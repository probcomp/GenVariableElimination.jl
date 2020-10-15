module GenVariableElimination

using Gen
using FunctionalCollections: PersistentSet, PersistentHashMap, dissoc, assoc, conj, disj

################
# factor graph #
################

struct VarNode{T,V} # T would be FactorNode, but for https://github.com/JuliaLang/julia/issues/269
    addr::Any
    factor_nodes::PersistentSet{T}
    idx_to_domain::Vector{V} # TODO need to populate these
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

struct FactorNode{N} # N is the number of variables in the (original?) factor graph
    var_nodes::Vector{VarNode} # immutable
    factor::Array{Float64,N} # immutable
end

vars(node::FactorNode) = node.var_nodes
factor(node::FactorNode) = node.factor

struct FactorGraph{N}
    var_nodes::PersistentHashMap{Any,VarNode}
    factor_nodes::PersistentSet{FactorNode{N}} # TODO not needed?
    addr_to_idx::PersistentHashMap{Any,Int}
    idx_to_addr::Vector{Any} # TODO
end

# variable elimination
# - generates a sequence of factor graphs
# - multiply all factors that mention the variable, generating a product factor, which replaces the other factors
# - then sum out the product factor, and remove the variable
# ( we could break these into two separate operations -- NO )

# all factors are of the same dimension, but with singleton dimensions for
# variables that are eliminated

function multiply_and_sum(factors::Array{Float64,N}, idx_to_sum_over::Int) where {N}
    result = copy(factors[1])
    for factor in factors[2:end]
        # note: this uses broadcasting of singleton dimensions
        result = result .* factor # TODO do it in place or using operator fusion
    end
    return sum(result, dims=idx_to_sum_over)
end

function eliminate(fg::FactorGraph{N}, addr::Any) where{N}
    eliminated_var_node = fg.var_nodes[addr]
    new_factor_nodes = fg.factor_nodes
    factors_to_combine = Vector{Array{Float64,N}}()
    other_involved_var_nodes = Dict{Any,VarNode{FactorNode{N}}}()
    for factor_node in factor_nodes(eliminated_var_node)
        push!(factors_to_combine, factor(factor_node))

        # remove the factor node
        new_factor_nodes = disj(new_factor_nodes, factor_node)

        # remove the reference to this factor node from its variable nodes
        for other_var_node::VarNode{FactorNode{N}} in vars(factor_node)
            if !haskey(other_involved_var_nodes, addr(other_var_node))
                other_involved_var_nodes[addr(other_var_node)] = other_var_node
            else
                other_var_node = other_involved_var_nodes[addr(other_var_node)]
            end
            @assert factor_node in factors(other_var_node)
            other_var_node = remove_factor_node(other_var_node, factor_node)
            other_involved_var_nodes[addr(other_var_node)] = other_var_node
        end
    end

    # compute the new factor
    # TODO use log space
    new_factor = multiply_and_sum(factors_to_combine, fg.addr_to_idx[addr])

    # add the new factor node
    new_factor_node = FactorNode{N}(var_nodes_for_new_factor, new_factor)
    new_factor_nodes = conj(new_factor_nodes, new_factor_node)
    for (a, other_var_node) in var_nodes_for_new_factor
        var_nodes_for_new_factor[a] = add_factor_node(other_var_node, new_factor_node)
    end

    # remove the eliminated var node
    new_var_nodes = dissoc(fg.var_nodes, eliminated_var_node)

    # replace old other var nodes with new other var nodes
    for (a, other_var_node) in var_nodes_for_new_factor
        new_var_nodes = assoc(new_var_nodes, a, other_var_node)
    end

    return FactorGraph{N}(new_var_nodes, new_factor_nodes, fg.addr_to_idx)
end

function conditional_dist(fg::FactorGraph{N}, other_values::Dict{Any,Any}, addr::Any)

    # TODO finish

    # other_values must contain a value for all variables that have a factor in
    # common with variable addr in fg
    var_node = fg.var_nodes[addr]
    n = num_values(var_node)
    # TODO use log space
    probs = ones(n)
    value_idx_vector = Vector{Int}(undef, N)
    for node_idx in 1:N
        other_addr = node_idx_to_addr(fg, node_idx)
        idx_vector[node_idx] = value_to_idx(fg.var_nodes[other_addr], other_values[other_addr])
    end
    for factor_node in factor_nodes(var_node)
        @assert factor_node in fg.factor_nodes # TODO not needed?
        f = factor(factor_node)
        for idx in 1:n
            CartesianIndex{N}(
            f[,idx]
        end
    end
end

function simulate_and_logpdf_addr(fg::FactorGraph{N}, other_values::Dict{Any,Any}, addr::Any)
    dist = conditional_dist(fg, other_values, addr)
    idx = categorical(dist)
    value = idx_to_value(fg.var_nodes[addr], idx)
    return (value, log(dist[idx]))
end

function logpdf_addr(fg::FactorGraph{N}, other_values::Dict{Any,Any}, addr::Any, value:Any)
    dist = conditional_dist(fg, other_values, addr)
    idx = value_to_idx(fg.var_nodes[addr], value)
    return log(dist[idx])
end

function sample_and_log_prob(fg::FactorGraph{N}, elimination_order) where {N}
    addr_to_fg = Dict{Any,FactorGraph{N}}()
    for addr in elimination_order
        addr_to_fg[addr] = fg
        fg = eliminate(fg, addr)
    end
    values = Dict{Any,Any}()
    total_log_prob = 0.0
    for addr in reverse(elimination_order)
        fg = addr_to_fg[addr]
        (values[addr], log_prob) = simulate_and_logpdf_addr(fg, values, addr)
        total_log_prob += log_prob
    end
    return (values, total_log_prob)
end

function log_prob(fg::FactorGraph{N}, values::Dict{Any,Any}, elimination_order)
    addr_to_fg = Dict{Any,FactorGraph{N}}()
    for addr in elimination_order
        addr_to_fg[addr] = fg
        fg = eliminate(fg, addr)
    end
    values = Dict{Any,Any}()
    total_log_prob = 0.0
    for addr in reverse(elimination_order)
        fg = addr_to_fg[addr]
        log_prob = logpdf_addr(fg, values, addr, values[addr])
        total_log_prob += log_prob
    end
    return total_log_prob
end

# a factor graph contains:
# - a map from address to variable node
# - a set of factor nodes, which may be initially present or constructed during VE

# methods on factor graphs:
# - log joint probability of a complete assignment to all variables in the FG (used, at minimum, during sampling)

# needed:
# > get a list of all factors that mention a variable
# > generate the product factor
# > sum out a product factor
# > remove factors and add the new summed out factor

# sampling and joint probability given variable elimination sequence - 
# in reverse elimination order:
# - we have a partial assignment to all variables that came after the variable in the elimination ordering
# - look up the appropriate intermediate factor graph, which is the FG immediately before eliminating the variable
#   identify all factors in this intermediate FG that are connected to this variable
#   take the relevant slices for each factor, resulting in vectors, and point-wise multiply them
#   normalize, and then sample from this distribution, and add the value of the sample to the partial assignment

# joint probability given variable elimination sequence and full assignment
# - do the same process as above, but taking values instead of sampling them

# constructor from trace and addr info (queries trace with update)

struct AddrInfo{T,U}
    domain::Vector{T}
    parent_addrs::Vector{U}
end


function to_factor_graph(trace, info::Dict{Any,AddrInfo})
    var_nodes_dict = Dict{Any,VarNode}()
    factor_nodes_dict = Dict{Any,FactorNode}()

    # TODO finish

    for (addr, _) in info
        var_nodes_dict[addr] = VarNode(addr, FactorNode[])
        factor_nodes_dict[addr] = FactorNode(addr, VarNode[], nothing)
    end

    for (addr, addr_info) in info
        node = var_nodes_dict[addr]
        for parent_addr in addr_info.parent_addrs
            parent_node = var_nodes_dict[parent_addr]
            push!(node.parents, parent_node)
            push!(parent_node.children, node)
        end
    end

    # populate factors with values by probing trace with update
    # TODO finish
    for (addr, addr_info) in info
        factor = factor_nodes_dict[addr].factor # Array{Float64}, multi-dimensional array
        for (idx, addrs, values_tuple) in enumerate_values()
            (_, weight, _, _) = update(trace, _, _, choices)
            factor[idx] = weight
        end
        @assert !isnan(any(factor))
    end

    return FactorGraph(collect(factor_nodes_dict), collect(var_nodes_dict))
end

###############################################
# step 1: compile a trace into a factor graph #
###############################################

@gen function foo()
    x ~ bernoulli(0.5)
    y ~ bernoulli(x ? 0.1 : 0.9)
    z ~ bernoulli((x && y) ? 0.4 : 0.9)
    w ~ bernoulli(z ? 0.4 : 0.5)
end

info = Dict{Any,AddrInfo}()
info[:x] = AddrInfo([true, false], [])
info[:y] = AddrInfo([true, false], [:x])
info[:z] = AddrInfo([true, false], [:x, :y])
info[:w] = AddrInfo([true, false], [:z])

trace = simulate(foo, ())

fg = to_factor_graph(trace, info)

########################################################
# step 2: run variable elimination in the factor graph #
########################################################

# each time you eliminate a variable X, you:
# - create a new factor that contains X and all of the variables with which it appears in factors
# - then we eliminate X, by replacing it with a new factor that contians all of these variables except for X
# - 

# elimination is a sequence of factor graphs...
# use FunctionalCollections?
eliminate!(factor_graph, :w)
eliminate!(factor_graph, :z)
eliminate!(factor_graph, :x)

# you end up with ... 


##############################################
# step 3: sample from the joint distribution #
##############################################

# variables get sampled in the reverse order from which they were eliminated
# we use the intermediate factor graph right before they are eliminated to
# compute the distribution

# (we just enumerate over the value of the variable to be sampled, and compute
# the total potential of the factor graph, for each value --- where all other
# variables in the FG have their values instantiated)

function joint_sample(factor_graph, elimination)

    # TODO
    # return the joint sample, and the log joint probability
end

function joint_evaluate(factor_graph, elimination, values)

    # TODO
    # return the log joint probability
end


############################################
# step 4: wrap it in a generative function #
############################################

# it's okay if the only operations supported by the generative function are
# simulate(), and generate() given all values

# idea: the bayesian network could actually not match exactly
# and also the enumeration grid could actually not match exactly either
# and the move will still be valid, since it is within MH
# (and -- we can check that it's valid by asserting that it accepts)
end # module
