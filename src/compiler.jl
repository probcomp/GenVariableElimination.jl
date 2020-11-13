using FunctionalCollections: PersistentSet, PersistentHashMap, dissoc, assoc, conj, disj
using Gen

#########################################
# compiling a trace into a factor graph #
#########################################

struct Latent{T,U}
    domain::Vector{T}
    parent_addrs::Vector{U}
end

struct Observation{U}
    parent_addrs::Vector{U} # only include addrs that are selected
end

parent_addrs(info::Union{Latent,Observation}) = info.parent_addrs

function get_domain_to_idx(domain::Vector{T}) where {T}
    domain_to_idx = Dict{T,Int}()
    for (i, value) in enumerate(domain)
        domain_to_idx[value] = i
    end
    return domain_to_idx
end

function cartesian_product(value_lists)
    tuples = Vector{Tuple}()
    for value in value_lists[1]
        if length(value_lists) > 1
            append!(tuples,
                [(value, rest...) for rest in cartesian_product(value_lists[2:end])])
        else
            append!(tuples, [(value,)])
        end
    end
    return tuples
end

# addr could be latent or obserrved...
# latent_addrs is the set of latent variables that are involved, which may or
# may not include the actual addr itself..

function create_factor(
        trace, addr, 
        latents::Dict{Any,Latent}, observations::Dict{Any,Observation},
        all_latent_addrs::Vector{Any})
    N = length(all_latent_addrs)
    in_factor = Vector{Bool}(undef, N)
    if haskey(latents, addr)
        for (i, a) in enumerate(all_latent_addrs)
            in_factor[i] = (a == addr || a in parent_addrs(latents[addr]))
        end
        num_vars = length(parent_addrs(latents[addr]))+1
    elseif haskey(observations, addr)
        for (i, a) in enumerate(all_latent_addrs)
            in_factor[i] = (a in parent_addrs(observations[addr]))
        end
        num_vars = length(parent_addrs(observations[addr]))
    end
    dims = map(i -> in_factor[i] ? length(latents[all_latent_addrs[i]].domain) : 1, 1:N)
    log_factor = Array{Float64,N}(undef, dims...)
    view_inds = map(i -> in_factor[i] ? Colon() : 1, 1:N)
    log_factor_view = view(log_factor, view_inds...)
    var_addrs = Vector{Any}(undef, num_vars)
    value_idx_lists = Vector{Any}(undef, num_vars)
    j = 1
    for i in 1:N
        if in_factor[i]
            a = all_latent_addrs[i]
            var_addrs[j] = a
            value_idx_lists[j] = collect(1:length(latents[a].domain))
            j += 1
        end
    end
    @assert j == num_vars + 1

    # populate factor with values by probing trace with update
    # the key idea is that this scales exponentially in maximum number of
    # parents of a variable, not the total number of variables

    cprod = cartesian_product(value_idx_lists)
    for value_idx_tuple in cprod
        choices = choicemap()
        for (a, value_idx) in zip(var_addrs, value_idx_tuple)
            choices[a] = latents[a].domain[value_idx]
        end
        (tmp_trace, _, _, _) = update(trace, get_args(trace), map((_)->NoChange(),get_args(trace)), choices)
        # NOTE: technically, the generative function of trace can use any
        # internal proposal, not only forward sampling, but this code is only
        # correct if the internal proposal uses forward sampling. enhancing the
        # trace interface with some more methods that specifically assume a
        # dependency graph would resolve this
        weight = project(tmp_trace, select(addr))
        log_factor_view[value_idx_tuple...] = weight
    end

    log_factor = log_factor .- logsumexp(log_factor[:])

    return (log_factor, var_addrs)
end

function compile_trace_to_factor_graph(
        trace, latents::Dict{Any,Latent}, observations::Dict{Any,Observation})

    # choose order of addresses (note, this is NOT the elimination order)
    # TODO does the order in which the addresses are indexed matter for e.g. cache performance? maybe?
    all_latent_addrs = collect(keys(latents))
    latent_addr_to_idx = Dict{Any,Int}()
    for (idx, addr) in enumerate(all_latent_addrs)
        latent_addr_to_idx[addr] = idx
    end

    # construct factor nodes, one for each latent and downstream variable
    N = length(all_latent_addrs)
    addr_to_factor_node = Dict{Any,FactorNode{N}}()
    factor_id = 1
    for addr in Iterators.flatten((keys(latents), keys(observations)))
        (log_factor, var_addrs) = create_factor(
            trace, addr, latents, observations, all_latent_addrs)
        addr_to_factor_node[addr] = FactorNode{N}(factor_id, Int[latent_addr_to_idx[a] for a in var_addrs], log_factor)
        factor_id += 1
    end
    num_factors = factor_id - 1

    # for each latent address, the set of addresses for factors that it is involved in
    children_and_self = [Set{Any}([all_latent_addrs[i]]) for i in 1:N]
    for (addr, addr_info) in Iterators.flatten((latents, observations))
        for parent_addr in parent_addrs(addr_info)
            push!(children_and_self[latent_addr_to_idx[parent_addr]], addr)
        end
    end

    # construct factor graph
    var_nodes = PersistentHashMap{Int,VarNode}()
    for (addr, addr_info) in latents
        i = latent_addr_to_idx[addr]
        factor_nodes = PersistentSet{FactorNode{N}}(
            [addr_to_factor_node[addr] for addr in children_and_self[i]])
        var_node = VarNode(addr, factor_nodes, addr_info.domain, get_domain_to_idx(addr_info.domain))
        var_nodes = assoc(var_nodes, latent_addr_to_idx[addr], var_node)
    end
    return FactorGraph{N}(num_factors, var_nodes, latent_addr_to_idx)
end

###########################################################
# generation of factor graph from static IR + combinators #
###########################################################

function factor_graph_analysis(
        trace::StaticIRTrace, addrs, cur_namespace, arg_ancestor_addrs::Vector{Set{Any}})
    ir = Gen.get_ir(Gen.get_gen_fn_type(typeof(trace)))
    node_to_ancestor_addrs = Dict{Any,Set{Any}}()
    for (node, arg_ancestors) in zip(ir.arg_nodes, arg_ancestor_addrs)
        node_to_ancestor_addrs[node] = arg_ancestors
    end
    latents = Dict{Any,Latent}()
    observations = Dict{Any,Observation}()
    for node in ir.nodes
        if isa(node, Gen.TrainableParameterNode)
            node_to_ancestor_addrs[node] = Set{Any}()
        elseif isa(node, Gen.ArgumentNode)
            # already handled above during initialization
        elseif isa(node, Gen.JuliaNode)
            if length(node.inputs) == 0
                node_to_ancestor_addrs[node] = Set{Any}()
            else
                node_to_ancestor_addrs[node] = union((node_to_ancestor_addrs[n] for n in node.inputs)...)
            end
        elseif isa(node, Gen.RandomChoiceNode)
            this_addr = foldr(=>, [cur_namespace..., node.addr])
            if this_addr in addrs
                domain = [discrete_finite_support_overapprox(
                    node.dist, (getproperty(trace, Gen.get_value_fieldname(n)) for n in node.inputs)...)...]
                if length(node.inputs) == 0
                    latents[this_addr] = Latent(domain, Any[])
                else
                    latents[this_addr] = Latent(domain, Any[union((node_to_ancestor_addrs[n] for n in node.inputs)...)...])
                end
                node_to_ancestor_addrs[node] = Set{Any}([this_addr]) # its value only depends on itself..
            else
                if length(node.inputs) == 0
                    observations[this_addr] = Observation(Any[])
                else
                    ancestor_addrs = union((node_to_ancestor_addrs[n] for n in node.inputs)...)
                    if !isempty(ancestor_addrs)
                        observations[this_addr] = Observation(Any[ancestor_addrs...])
                    end
                end
                node_to_ancestor_addrs[node] = Set{Any}()
            end
        elseif isa(node, Gen.GenerativeFunctionCallNode)
            (node_to_ancestor_addrs[node], call_latents, call_observations) = factor_graph_analysis(
                Gen.static_get_subtrace(trace, Val(node.addr)),
                addrs, (cur_namespace..., node.addr),
                Set{Any}[node_to_ancestor_addrs[n] for n in node.inputs])
            merge!(latents, call_latents)
            merge!(observations, call_observations)
        else
            @assert false
        end
    end
    return (node_to_ancestor_addrs[ir.return_node], latents, observations)
end

function factor_graph_analysis(trace::Gen.VectorTrace{Gen.UnfoldType}, addrs, cur_namespace, arg_ancestor_addrs::Vector{Set{Any}})
    gen_fn = get_gen_fn(trace)
    kernel = gen_fn.kernel
    n_ancestor_addrs = arg_ancestor_addrs[1]
    init_state_ancestor_addrs = arg_ancestor_addrs[2]
    params_ancestor_addrs = arg_ancestor_addrs[3:end]
    !isempty(n_ancestor_addrs) && error("the selected addresses may not modify the structure of the trace")

    # NOTE: we do an analysis of the inner static generative function, then we copy the results?
    # or a few analyses, interestingly..

    # note: we could require them to do 'All' here, just to simplify the analysis?
    # but this is overly restrictive..

    # TODO: Note that this analysis can be done without visiting all nodes
    # (this is interetsingly related to fixed points)

    if length(trace.subtraces) == 0
        return (Set{Any}(), Dict{Any,Latent}(), Dict{Any,Observation}())
    end
    
    node_to_ancestor_addrs = Dict{Int,Set{Any}}()
    latents = Dict{Any,Latent}()
    observations = Dict{Any,Observation}()
    prev_state_ancestor_addrs = init_state_ancestor_addrs

    for t in 1:length(trace.subtraces)
        (node_to_ancestor_addrs[t], call_latents, call_observations) = factor_graph_analysis(
                    trace.subtraces[t],
                    addrs, (cur_namespace..., t),
                    [Set{Any}(), prev_state_ancestor_addrs, params_ancestor_addrs...])
        prev_state_ancestor_addrs = node_to_ancestor_addrs[t]
        merge!(latents, call_latents)
        merge!(observations, call_observations)
    end

    return (union(values(node_to_ancestor_addrs)...), latents, observations)
end

function factor_graph_analysis(trace::Gen.VectorTrace{Gen.MapType}, addrs, cur_namespace, arg_ancestor_addrs::Vector{Set{Any}})
    # TODO
end

function factor_graph_analysis(trace, addrs)
    return factor_graph_analysis(trace, addrs, (), Set{Any}[Set{Any}() for _ in get_args(trace)])
end

@dist labeled_cat(labels, probs) = labels[categorical(probs)]

@gen function ve_backwards_sampler(latents, fg, ve_result)
    N = length(latents)
    values = Vector{Any}(undef, N)
    for addr in reverse(ve_result.elimination_order)
        var_idx = addr_to_idx(fg, addr)
        fg = ve_result.intermediate_fgs[var_idx]
        dist = conditional_dist(fg, values, addr)
        value = ({addr} ~ labeled_cat(latents[addr].domain, dist))
        values[var_idx] = value
    end
end

# NOTE: we could stage the computation better, so less is done within the
# generative function at runtime
# NOTE: could also generate one that was specialized to the specific values in this trace?
# NOTE: we could emit static IR instead of DML code

"""
    sampler::GenerativeFunction = generate_backwards_sampler_fixed_trace(trace, addrs)

Generate a generative function that takes no arguments that samples from the conditional distribution on the given addresses.

The addresses must be discrete random choices within finite support and must not affect the control flow in the trace.
The sampler is generated using variable elimination followed by backwards sampling, where the order of the provided addresses defines the elimination order.
The sampler takes no arguments and is specialized to the conditional distribution for the given trace.
"""
function generate_backwards_sampler_fixed_trace(trace, addrs)
    (_, latents, observations) = factor_graph_analysis(trace, addrs)
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    ve_result = variable_elimination(fg, addrs)
    @gen function sampler()
        {*} ~ ve_backwards_sampler(latents, fg, ve_result)
    end
    return sampler
end

"""
    sampler::GenerativeFunction = generate_backwards_sampler(trace, addrs)

Generate a generative function that takes a trace argument that samples from the conditional distribution on the given addresses.

The addresses must be discrete random choices within finite support and must not affect the control flow in the trace.
The sampler is generated using variable elimination followed by backwards sampling, where the order of the provided addresses defines the elimination order.
The sampler is specialized to traces with the same control flow path, but not necessarily the same values, as the trace provided to the generation function.
"""
function generate_backwards_sampler(trace, addrs)
    (_, latents, observations) = factor_graph_analysis(trace, addrs)
    @gen function sampler(trace)
        fg = compile_trace_to_factor_graph(trace, latents, observations)
        ve_result = variable_elimination(fg, addrs)
        {*} ~ ve_backwards_sampler(latents, fg, ve_result)
    end
    return sampler
end
