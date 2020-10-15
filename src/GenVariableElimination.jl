module GenVariableElimination

using Gen
using FunctionalCollections: PersistentSet, PersistentHashMap, dissoc, assoc, conj, disj

################
# factor graph #
################

struct VarNode{T} # T would be FactorNode, but for https://github.com/JuliaLang/julia/issues/269
    factor_nodes::PersistentSet{T}
end

factor_nodes(node::VarNode) = node.factor_nodes

struct FactorNode{N} # N is the number of variables in the (original?) factor graph
    #var_nodes::Vector{VarNode} # immutable -- needed?
    factor::Array{Float64,N} # immutable
end

factor(node::FactorNode) = node.factor

struct FactorGraph{N}
    var_nodes::PersistentHashMap{Any,VarNode}
    factor_nodes::PersistentSet{FactorNode{N}}
    addr_to_idx::Dict{Any,Int}
end

# maybe all factors can be of the same dimension, but with singleton dimensions for variables that are eliminated
# that would make indexing much easier

function multiply_and_sum(factors::Array{Float64,N}, idx_to_sum_over::Int) where {N}
    result = copy(factors[1])
    for factor in factors[2:end]
        result = result .* factor # TODO do it in place
    end
    return sum(result, dims=idx_to_sum_over)
end

function eliminate(fg::FactorGraph{N}, addr::Any) where{N}
    var_node = fg.var_nodes[addr]
    new_factor_nodes = fg.factor_nodes
    factors_to_combine = Vector{Array{Float64,N}}()
    for factor_node in factor_nodes(var_node)
        new_factor_nodes = disj(new_factor_nodes, factor_node)
        push!(factors_to_combine, factor(factor_node))
    end
    new_factor = multiply_and_sum(factors_to_combine, fg.addr_to_idx[addr])
    new_factor_node = FactorNode{N}(new_factor)
    new_var_nodes = dissoc(fg.var_nodes, var_node)
    new_fg = FactorGraph{N}(new_var_nodes, new_factor_nodes, fg.addr_to_idx)
    return new_fg
end

# a factor graph contains:
# - a map from address to variable node
# - a set of factor nodes, which may be initially present or constructed during VE

# methods on factor graphs:
# - log joint probability of a complete assignment to all variables in the FG (used, at minimum, during sampling)

# variable elimination
# - generates a sequence of factor graphs
# - new_fg = eliminate(fg, address) 
# - multiply all factors that mention the variable, generating a product factor, which replaces the other factors
# - then sum out the product factor, and remove the variable
# ( we could break these into two separate operations -- NO )

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

###############################################
# step 1: compile a trace into a factor graph #
###############################################


struct FactorGraph
    factor_nodes::Vector{FactorNode}
    var_nodes::Vector{VarNode}
    addr_to_var_idx::Dict{Any,Int}# TODO
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

###########
# example #
###########

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

# idea: the bayesian network could actually not match exactly
# and also the enumeration grid could actually not match exactly either
# and the move will still be valid, since it is within MH
# (and -- we can check that it's valid by asserting that it accepts)
end # module
