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

    for value_idx_tuple in cartesian_product(value_idx_lists)
        choices = choicemap()
        for (a, value_idx) in zip(var_addrs, value_idx_tuple)
            choices[a] = latents[a].domain[value_idx]
        end
        (tmp_trace, _, _, _) = update(trace, get_args(trace), map((_)->NoChange(),get_args(trace)), choices)
        # NOTE: semantics of project not exactly aligned -- since project can use any proposal...
        weight = project(tmp_trace, select(addr))
        log_factor_view[value_idx_tuple...] = weight
    end

    factor = exp.(log_factor .- logsumexp(log_factor[:])) # TODO shift the rest of the code to work in log space

    return (factor, var_addrs)
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
        (factor, var_addrs) = create_factor(
            trace, addr, latents, observations, all_latent_addrs)
        addr_to_factor_node[addr] = FactorNode{N}(factor_id, Int[latent_addr_to_idx[a] for a in var_addrs], factor)
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
