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

###################
# static analysis #
###################

#function generate_factor_graph_template(trace, addrs, domains)
    # generate the factor graph, but don't populate the factors yet...
    # this could involve generating a Julia function for each factor
    # that computes its value from the trace?
#end 

# this is the key thing that needs to happen efficiently...

# note that factors can cross boundaries of generative functions..
# e.g. a static IR function that calls two other Static IR functions
# (or, e.g. unfold)

# perhaps the first step is to generate a flat Bayesian network representation?
# (where only latent variables and downstream variables are included?)

# and then generate the factor graph from that..

#function generate_flat_bayesian_network(trace, addrs)
    # each node will need to be labeled with its addr (which also defines
    # where in the IR computation graph / trace it is..)
    # Q: if you return multiple outputs, are the independencies tracked separately?
    # the nodes should also be arranged hierarchically?

    # we need to compute the set of observed random variables, and their parents..
    # this requires an inter-procedural graph analysis, where we propagate
    # through the computation graph and stop at random choices (also
    # propagating the set of parents at each stage via union), then the set of
    # random choices that we stop at are the 'observations' for the purposes of
    # this analysis

    # we also need to determine what latent random variables depend on what others..

    # both of these can be achieved by a propagation along the
    # (inter-procedural) computation graph; which seems like a general operation..

    # 1. do propagation in the inter-procedural flat computation graph
    # 2. populate factors, or generate code for populating factors...
    #    (in general, this will involve
    #           (a) iterating over all values in the cartesian product of parents and
    #           (b) for each tuple of values, computing the arguments to the distribution, and
    #           (c) calling Gen.logpdf for each one
    # (b) is the challenging part..

    # after we have that information, we need to populate the factors
    # this involves enumerating over all the values of all of the parents
    # and re-executing small chunks of the SML function to compute logpdfs

    # maybe the Static IR (+combinators) should support a query of the form:
    # (i) get parents and get probability density of child given parents
    # we really need to enumerate over all values of the parents, and for each one,
    # query the probability of the child

    # the purpose of this is to make it easier to generate the factors..

    # for the initial version, maybe we should use use the Static IR to compute the
    # latents and observations, and just use `update' and `project' as before?
    # we could also add a new (optional) method to the GFI which returns conditional densities given parents
    # would need to define its semantics..

#end

# NOTE:
# a big part of this is changing where the code goes.
# instead of all the code going with the modeling language or the combinatoars
# and instead of exposing a huge API
# we instead expose a simpler but much more flexible API (for the IR itself)
# and then all the code gets written outside in separate modules, instead of split up between modeling components themselves..

function forward_analysis(trace::StaticIRTrace, addrs, cur_namespace, arg_ancestor_addrs::Vector{Set{Any}})
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
                domain = [Gen.discrete_finite_support_overapprox(
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
            (node_to_ancestor_addrs[node], call_latents, call_observations) = forward_analysis(
                static_get_subtrace(trace, node.addr), 
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

function forward_analysis(trace::Gen.VectorTrace{Gen.UnfoldType}, addrs, arg_ancestor_addrs::Vector{Set{Any}})
    # TODO
end

function forward_analysis(trace::Gen.VectorTrace{Gen.MapType}, addrs, arg_ancestor_addrs::Vector{Set{Any}})
    # TODO
end

#function generate_factor_graph_info(trace::StaticIRTrace, addrs)

    #struct Latent{T,U}
        #domain::Vector{T}
        #parent_addrs::Vector{U}
    #end
    
    #struct Observation{U}
        #parent_addrs::Vector{U} # only include addrs that are selected
    #end

    # for each address, compute its domain (this is possible by finding hte Distribution that the address corresponds to)
    # for each address, identify the parents
    # identify the set of observation addresses



    # for each latent, we need: (i) its domain, and (ii) its parents
    # for each observation, we need: (ii) its parent

    # the set of observations and the parent sets can be generated by walking
    # the computation graph and propagating sets of parents
    # this probablty shares some logic with current parts of the Static IR
    # implementation, but would need to be inter-procedural..

    # we can do the analysis compositionally, walking along the call tree..
    # the input is, for each argument, the set of addresses (in top-level namespace) that it effects,
    # and the set of all latent addresses (so we can find our own)
    # the output is: (i) for the return value, the set of latent addresses that it depends on, if any
    #                (ii) a listing of the 'latents' and 'observations' within the function call
    # for SSA blocks, we just propagate forward (we don't need to propagate when a return value does not depend on any latent addrs)
    # for Map:
    #   Map(fn)(arg_vec1, arg_vec2, arg_vec3)
    #   for now, if any of the arguments changed, then we need to include factors for all applications
    #   and the return value may be marked, unless the kernel is absorbing..
    # for Unfold(fn)(n, init_state, param1, param2)
    #   for now, n cannot change (this should result in an error)
    #   also, if the params change then, we will need to potentially visit every kernel to identify observations?
    #   if the init_state changes, then we will need to ...
    # for Recurse
    # for Switch

    # later version: be able to track finer-grained information about the
    # dependences of different elements of collections, for now, not..
#end

@dist labeled_cat(labels, probs) = labels[categorical(probs)]

function generate_conditional_sampler(trace::StaticIRTrace, addrs)

    # NOTE: we could stage the computation better, so less is done within the generative function at runtime
    # NOTE: could also generate one that was specialized to the specific values in this trace?
    # NOTE: we could eventually emit static IR instead of DML code
    (_, latents, observations) = forward_analysis(trace, addrs, (), Set{Any}[Set{Any}() for _ in get_args(trace)])
 
    @gen function ve_sampler(trace)
        #latents, observations = generate_factor_graph_info(trace, addrs)
        fg = compile_trace_to_factor_graph(trace, latents, observations)
        elimination_result = variable_elimination(fg, addrs)
        N = length(latents)
        values = Vector{Any}(undef, N)
        for addr in reverse(elimination_result.elimination_order)
            var_idx = addr_to_idx(fg, addr)
            fg = elimination_result.intermediate_fgs[var_idx]
            dist = conditional_dist(fg, values, addr)
            value = ({addr} ~ labeled_cat(latents[addr].domain, dist))
            values[var_idx] = value
        end
    end   

    return ve_sampler
end
