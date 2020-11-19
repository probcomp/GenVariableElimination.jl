using FunctionalCollections: PersistentSet, PersistentHashMap, dissoc, assoc, conj, disj

####################################################
# factor graph, variable elimination, and sampling #
####################################################

# TODO performance optimize?

struct VarNode{T,V} # T would be FactorNode, but for https://github.com/JuliaLang/julia/issues/269
    addr::Any
    factor_nodes::PersistentSet{T}
    idx_to_domain::Vector{V}
    domain_to_idx::Dict{V,Int}
end

addr(node::VarNode) = node.addr
factor_nodes(node::VarNode) = node.factor_nodes
num_values(node::VarNode) = length(node.idx_to_domain)
idx_to_value(node::VarNode{T,V}, idx::Int) where {T,V} = node.idx_to_domain[idx]::V
value_to_idx(node::VarNode{T,V}, value::V) where {T,V} = node.domain_to_idx[value]

function remove_factor_node(node::VarNode{T,V}, factor_node::T) where {T,V}
    return VarNode{T,V}(
        node.addr, disj(node.factor_nodes, factor_node),
        node.idx_to_domain, node.domain_to_idx)
end

function add_factor_node(node::VarNode{T,V}, factor_node::T) where {T,V}
    return VarNode{T,V}(
        node.addr, conj(node.factor_nodes, factor_node),
        node.idx_to_domain, node.domain_to_idx)
end

struct Factor{M}
    data::Array{Float64,M}
    var_to_idx::Vector{Int}
    idx_to_var::Vector{Int}
end

struct FactorNode{N} # N is the number of variables in the (original) factor graph
    id::Int
    vars::Vector{Int} # immutable
    #log_factor::Array{Float64,N} # immutable
    factor::Factor
end

vars(node::FactorNode) = node.vars
#get_log_factor(node::FactorNode) = node.log_factor
get_factor(node::FactorNode) = node.factor

struct FactorGraph{N}
    num_factors::Int
    var_nodes::PersistentHashMap{Int,VarNode}

    # NOTE: when variables get eliminated from a factor graph, they don't get reindex]
    # (i.e. these fields are unchanged)
    addr_to_idx::Dict{Any,Int} 
end

# just for testing purposes:
function factor_value(fg::FactorGraph, node::FactorNode{N}, values::Dict) where {N}
    idxs = (idx in vars(node) ? value_to_idx(idx_to_var_node(fg, idx), values[idx]) : 1 for idx in 1:N)
    return exp.(node.log_factor[idxs...])
end

function draw_factor_graph(fg::FactorGraph, graphviz, fname, addr_to_name)
    dot = graphviz.Digraph()
    factor_idx = 1
    for node in values(fg.var_nodes)
        shape = "ellipse"
        color = "white"
        name = addr_to_name(addr(node))
        dot[:node](name, name, shape=shape, fillcolor=color, style="filled")
        for factor_node in factor_nodes(node)
            shape = "box"
            color = "gray"
            factor_name = string(factor_node.id)
            dot[:node](factor_name, factor_name, shape=shape, fillcolor=color, style="filled")
            dot[:edge](name, factor_name)
        end
    end
    dot[:render](fname, view=true)
end


idx_to_var_node(fg::FactorGraph, idx::Int) = fg.var_nodes[idx]
addr_to_idx(fg::FactorGraph, addr) = fg.addr_to_idx[addr]

# variable elimination
# - generates a sequence of factor graphs
# - multiply all factors that mention the variable, generating a product factor, which replaces the other factors
# - then sum out the product factor, and remove the variable
# ( we could break these into two separate operations -- NO )

# all factors are of the same dimension, but with singleton dimensions for
# variables that are eliminated

#function multiply_and_sum(log_factors::Vector{Array{Float64,N}}, idx_to_sum_over::Int) where {N}
    #result = broadcast(+, log_factors...)
    #m = maximum(result, dims=idx_to_sum_over)
    #return m .+ log.(sum(exp.(result .- m), dims=idx_to_sum_over))
#end

function fast_multiply_and_sum(factors, var_to_sum_over::Int, var_dims::Vector{Int})

    # the set of variables are those that appear in any factor
    # the order in which the variables appear in the factor is arbitrary. we choose one
    var_to_idx = fill(-1, length(var_dims))
    next_idx = 1
    for lf in factors
        for var in lf.idx_to_var
            if var_to_idx[var] < 0
                var_to_idx[var] = next_idx
                next_idx += 1
            end
        end
    end
    nvars = next_idx-1
    idx_to_var = Vector{Int}(undef, nvars)
    for (var, idx) in enumerate(var_to_idx)
        if idx > 0
            idx_to_var[idx] = var
        end
    end

    # once the order of the variables is chosen, we can compute the size of the new factor
    new_factor_size = Int[var_dims[var] for var in idx_to_var]

    # then initialize the new factor
    data = Array{Float64,nvars}(undef, new_factor_size...)

    # populate the new factor
    #  loop over entries in the new factor and compute them one by one, by summing
    var_to_sum_dim = var_dims[var_to_sum_over]
    var_vals = Vector{Int}(undef, nvars)
    for i in 1:length(factor)
        idx_vals = CartesianIndices(factor)[i] # map from the new log factor indices to values
        for var in 1:nvars
            var_vals[var] = idx_vals[idx_to_var[var]] # note: unecessary computation here..
        end
        total = 0.0
        for val in 1:var_to_sum_dim
            prod = 0.0
            for lf_idx in 1:length(factors)
                lf = factors[lf_idx]
                idx = CartesianIndex((var == var_to_sum_over ? val : var_vals[var] for var in lf.idx_to_var)...)
                prod *= lf.data[idx]
            end
            total += prod
        end
        data[i] = total
    end
    return Factor{nvars}(data, var_to_idx, idx_to_var)
end


function eliminate(fg::FactorGraph{N}, addr::Any) where{N}
    eliminated_var_idx = addr_to_idx(fg, addr)
    eliminated_var_node = idx_to_var_node(fg, eliminated_var_idx)
    log_factors_to_combine = Vector{Array{Float64,N}}()
    other_involved_var_nodes = Dict{Int,VarNode{FactorNode{N}}}()
    for factor_node in factor_nodes(eliminated_var_node)
        push!(log_factors_to_combine, get_log_factor(factor_node))

        # remove the reference to this factor node from its variable nodes
        for other_var_idx::Int in vars(factor_node)
            if other_var_idx == eliminated_var_idx
                continue
            end
            if !haskey(other_involved_var_nodes, other_var_idx)
                other_var_node = idx_to_var_node(fg, other_var_idx)
                other_involved_var_nodes[other_var_idx] = other_var_node
            else
                other_var_node = other_involved_var_nodes[other_var_idx]
            end
            @assert factor_node in factor_nodes(other_var_node)
            other_var_node = remove_factor_node(other_var_node, factor_node)
            other_involved_var_nodes[other_var_idx] = other_var_node
        end
    end

    # compute the new factor
    #new_log_factor = multiply_and_sum(log_factors_to_combine, eliminated_var_idx)
    var_dims = Int[num_values(fg.var_nodes[i]) for i in 1:length(fg.var_nodes)]
    new_factor = fast_multiply_and_sum(log_factors_to_combine, eliminated_var_idx, var_dims)

    # add the new factor node
    new_factor_node = FactorNode{N}(
        fg.num_factors+1, collect(keys(other_involved_var_nodes)), new_factor)#new_log_factor)
    for (other_var_idx, other_var_node) in other_involved_var_nodes
        other_involved_var_nodes[other_var_idx] = add_factor_node(other_var_node, new_factor_node)
    end

    # remove the eliminated var node
    new_var_nodes = dissoc(fg.var_nodes, eliminated_var_idx)

    # replace old other var nodes with new other var nodes
    for (other_var_idx, other_var_node) in other_involved_var_nodes
        new_var_nodes = assoc(new_var_nodes, other_var_idx, other_var_node)
    end

    return FactorGraph{N}(fg.num_factors+1, new_var_nodes, fg.addr_to_idx)
end

function conditional_dist(fg::FactorGraph{N}, values::Vector{Any}, addr::Any) where {N}

    # other_values must contain a value for all variables that have a factor in
    # common with variable addr in fg
    var_idx = addr_to_idx(fg, addr)
    var_node = idx_to_var_node(fg, var_idx)
    n = num_values(var_node)
    #println("conditional_dist, addr: $addr, var_idx: $var_idx, num_values: $n")
    log_probs = zeros(n)
    # TODO : writing the slow version first..
    # LATER: use generated function to generate a version that is specialized
    # to N? (unroll this loop, and inline the indices..)
    #indices = Vector{Int}(undef, N)
    for i in 1:n
        for factor_node in factor_nodes(var_node)
            #log_factor::Array{Float64,N} = get_log_factor(factor_node)
            #fill!(indices, 1)
            #for other_var_idx in vars(factor_node)
                #if other_var_idx != var_idx
                    #other_var_node = idx_to_var_node(fg, other_var_idx)
                    #indices[other_var_idx] = value_to_idx(other_var_node, values[other_var_idx])
                #end
            #end
            #indices[var_idx] = i
            #log_probs[i] += log_factor[CartesianIndex{N}(indices...)]
            factor = get_factor(factor_node)
            indices = Int[value_to_idx(values[var]) for var in factor.idx_to_var]
            log_probs[i] += log(factor.data[CartesianIndex(indices...)])
        end
    end
    return exp.(log_probs .- logsumexp(log_probs))
end

struct VariableEliminationResult{N}
    elimination_order::Any
    intermediate_fgs::Vector{FactorGraph{N}}
end

function variable_elimination(fg::FactorGraph{N}, elimination_order) where {N}
    intermediate_fgs = Vector{FactorGraph{N}}(undef, N)
    for addr in elimination_order
        var_idx = addr_to_idx(fg, addr)
        intermediate_fgs[var_idx] = fg
        fg = eliminate(fg, addr)
    end
    return VariableEliminationResult(elimination_order, intermediate_fgs)
end
