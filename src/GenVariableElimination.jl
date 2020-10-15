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
        end
    end
end

function sample_and_compute_log_prob_addr(fg::FactorGraph{N}, other_values::Dict{Any,Any}, addr::Any)
    dist = conditional_dist(fg, other_values, addr)
    idx = categorical(dist)
    value = idx_to_value(fg.var_nodes[addr], idx)
    return (value, log(dist[idx]))
end

function compute_log_prob_addr(fg::FactorGraph{N}, other_values::Dict{Any,Any}, addr::Any, value:Any)
    dist = conditional_dist(fg, other_values, addr)
    idx = value_to_idx(fg.var_nodes[addr], value)
    return log(dist[idx])
end

function sample_and_compute_log_prob(fg::FactorGraph{N}, elimination_order) where {N}
    addr_to_fg = Dict{Any,FactorGraph{N}}()
    for addr in elimination_order
        addr_to_fg[addr] = fg
        fg = eliminate(fg, addr)
    end
    values = Dict{Any,Any}()
    total_log_prob = 0.0
    for addr in reverse(elimination_order)
        fg = addr_to_fg[addr]
        (values[addr], log_prob) = sample_and_compute_log_prob_addr(fg, values, addr)
        total_log_prob += log_prob
    end
    return (values, total_log_prob)
end

function compute_log_prob(fg::FactorGraph{N}, elimination_order, values::Dict{Any,Any})
    addr_to_fg = Dict{Any,FactorGraph{N}}()
    for addr in elimination_order
        addr_to_fg[addr] = fg
        fg = eliminate(fg, addr)
    end
    values = Dict{Any,Any}()
    total_log_prob = 0.0
    for addr in reverse(elimination_order)
        fg = addr_to_fg[addr]
        log_prob = compute_log_prob_addr(fg, values, addr, values[addr])
        total_log_prob += log_prob
    end
    return total_log_prob
end

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

####################################
# wrap it in a generative function #
####################################

struct FactorGraphSamplerTrace
    fg::FactorGraph
    args::Tuple{Gen.Trace,Dict{Any,AddrInfo},Any}
    choices::Gen.DynamicChoiceMap
    log_prob::Float64
end

struct FactorGraphSampler <: GenerativeFunction{Nothing,FactorGraphSamplerTrace}
end

const compile_and_sample_factor_graph = FactorGraphSampler()

function Gen.simulate(gen_fn::FactorGraphSampler, args::Tuple)
    (model_trace, info, elimination_order) = args
    fg = compile_factor_graph(model_trace, info)
    (values, log_prob) = sample_and_compute_log_prob(fg, elimination_order)
    choices = choicemap()
    for (addr, value) in values
        choices[addr] = value
    end
    return FactorGraphSamplerTrace(fg, args, choices, log_prob)
end

function Gen.generate(gen_fn::FactorGraphSampler, args::Tuple, choices::ChoiceMap)
    (model_trace, info, elimination_order) = args
    fg = compile_factor_graph(model_trace, info)
    values = Dict{Any,Any}()
    for addr in keys(info)
        values[addr] = choices[addr]
    end
    log_prob = compute_log_prob(fg, elimination_order, values)
    trace = FactorGraphSamplerTrace(fg, args, choices, log_prob)
    return (trace, log_prob)
end

Gen.get_args(trace::FactorGraphSampler) = trace.args
Gen.get_retval(trace::FactorGraphSampler) = nothing
Gen.get_choices(trace::FactorGraphSampler) = trace.values
Gen.get_score(trace::FactorGraphSampler) = trace.log_prob
Gen.get_gen_fn(trace::FactorGraphSampler) = compile_and_sample_factor_graph
Gen.project(trace::FactorGraphSampler, ::EmptyChoiceMap) = 0.0
Gen.has_argument_grads(gen_fn::FactorGraphSampler) = (false, false, false)
Gen.accepts_output_grad(gen_fn::FactorGraphSampler) = false

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

end # module
