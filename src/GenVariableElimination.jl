module GenVariableElimination
using Gen
using FunctionalCollections: PersistentSet, PersistentHashMap, dissoc, assoc, conj, disj
using PyCall

####################################################
# factor graph, variable elimination, and sampling #
####################################################

# TODO use logspace in probability calculations
# TODO performance optimize?
# TODO simplify FactorGraph data structure?

# TODO make the generative function accept the precompiled factor graph (with elimination already run), not the trace
# that way we can sample multiple times without having to re-run compilation or variable elimination each time

struct VarNode{T,V} # T would be FactorNode, but for https://github.com/JuliaLang/julia/issues/269
    addr::Any
    factor_nodes::PersistentSet{T}
    idx_to_domain::Vector{V}
    domain_to_idx::Dict{V,Int}
end

function VarNode(
        addr, factor_nodes::PersistentSet{T}, idx_to_domain::Vector{V},
        domain_to_idx::Dict{V,Int}) where {T,V}
    return VarNode{T,V}(addr, factor_nodes, idx_to_domain, domain_to_idx)
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

struct FactorNode{N} # N is the number of variables in the (original) factor graph
    id::Int
    vars::Vector{Int} # immutable
    factor::Array{Float64,N} # immutable
end

vars(node::FactorNode) = node.vars
factor(node::FactorNode) = node.factor

struct FactorGraph{N}
    num_factors::Int
    var_nodes::PersistentHashMap{Int,VarNode}

    # NOTE: when variables get eliminated from a factor graph, they don't get reindexed
    # (i.e. these fields are unchanged)
    addr_to_idx::Dict{Any,Int} 
end

# just for testing purposes:
function factor_value(fg::FactorGraph, node::FactorNode{N}, values::Dict) where {N}
    idxs = (idx in vars(node) ? value_to_idx(idx_to_var_node(fg, idx), values[idx]) : 1 for idx in 1:N)
    return node.factor[idxs...]
end

function draw_graph(fg::FactorGraph, graphviz, fname)
    dot = graphviz.Digraph()
    factor_idx = 1
    for node in values(fg.var_nodes)
        shape = "ellipse"
        color = "white"
        name = addr(node)
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

export draw_graph

idx_to_var_node(fg::FactorGraph, idx::Int) = fg.var_nodes[idx]
addr_to_idx(fg::FactorGraph, addr) = fg.addr_to_idx[addr]

# variable elimination
# - generates a sequence of factor graphs
# - multiply all factors that mention the variable, generating a product factor, which replaces the other factors
# - then sum out the product factor, and remove the variable
# ( we could break these into two separate operations -- NO )

# all factors are of the same dimension, but with singleton dimensions for
# variables that are eliminated

function multiply_and_sum(factors::Vector{Array{Float64,N}}, idx_to_sum_over::Int) where {N}
    # TODO debug / optimize
    result = copy(factors[1])
    for factor in factors[2:end]
        # note: this uses broadcasting of singleton dimensions
        result = result .* factor # TODO do it in place or using operator fusion
    end
    # produces an N-array with the summed out dimension having length 1
    return sum(result, dims=idx_to_sum_over) 
end

function eliminate(fg::FactorGraph{N}, addr::Any) where{N}
    eliminated_var_idx = addr_to_idx(fg, addr)
    eliminated_var_node = idx_to_var_node(fg, eliminated_var_idx)
    factors_to_combine = Vector{Array{Float64,N}}()
    other_involved_var_nodes = Dict{Int,VarNode{FactorNode{N}}}()
    for factor_node in factor_nodes(eliminated_var_node)
        push!(factors_to_combine, factor(factor_node))

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
    # TODO use log space
    new_factor = multiply_and_sum(factors_to_combine, eliminated_var_idx)

    # add the new factor node
    new_factor_node = FactorNode{N}(
        fg.num_factors+1, collect(keys(other_involved_var_nodes)), new_factor)
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
    println("conditional dist, addr: $addr")

    # other_values must contain a value for all variables that have a factor in
    # common with variable addr in fg
    var_idx = addr_to_idx(fg, addr)
    var_node = idx_to_var_node(fg, var_idx)
    n = num_values(var_node)
    probs = ones(n)
    # TODO : writing the slow version first..
    # LATER: use generated function to generate a version that is specialized to N (unroll this loop, and inline the indices..)
    indices = Vector{Int}(undef, N)
    for i in 1:n
        for factor_node in factor_nodes(var_node)
            F::Array{Float64,N} = factor(factor_node)
            fill!(indices, 1)
            for other_var_idx in vars(factor_node)
                if other_var_idx != var_idx
                    other_var_node = idx_to_var_node(fg, other_var_idx)
                    indices[other_var_idx] = value_to_idx(other_var_node, values[other_var_idx])
                end
            end
            probs[i] = probs[i] * F[CartesianIndex{N}(indices...)]
        end
    end
    return probs / sum(probs)
end

function sample_and_compute_log_prob_addr(fg::FactorGraph, values::Vector{Any}, addr::Any)
    println("sample and compute log prob addr: $addr")
    dist = conditional_dist(fg, values, addr)
    idx = categorical(dist)
    value = idx_to_value(idx_to_var_node(fg, addr_to_idx(fg, addr)), idx)
    return (value, log(dist[idx]))
end

function compute_log_prob_addr(fg::FactorGraph, values::Vector{Any}, addr::Any, value::Any)
    println("compute log prob addr: $addr")
    dist = conditional_dist(fg, values, addr)
    idx = value_to_idx(idx_to_var_node(fg, addr_to_idx(fg, addr)), value)
    return log(dist[idx])
end

function sample_and_compute_log_prob(fg::FactorGraph{N}, elimination_order) where {N}
    intermediate_fgs = Vector{FactorGraph{N}}(undef, N)
    for addr in elimination_order
        var_idx = addr_to_idx(fg, addr)
        intermediate_fgs[var_idx] = fg
        fg = eliminate(fg, addr)
    end
    values = Vector{Any}(undef, N)
    total_log_prob = 0.0
    for addr in reverse(elimination_order)
        var_idx = addr_to_idx(fg, addr)
        fg = intermediate_fgs[var_idx]
        (values[var_idx], log_prob) = sample_and_compute_log_prob_addr(fg, values, addr)
        total_log_prob += log_prob
    end
    addr_to_value = Dict{Any,Any}()
    for addr in elimination_order
        addr_to_value[addr] = values[addr_to_idx(fg, addr)]
    end
    return (addr_to_value, total_log_prob)
end

function compute_log_prob(fg::FactorGraph{N}, elimination_order, addr_to_value::Dict{Any,Any}) where {N}
    intermediate_fgs = Vector{FactorGraph{N}}(undef, N)
    for addr in elimination_order
        var_idx = addr_to_idx(fg, addr)
        intermediate_fgs[var_idx] = fg
        fg = eliminate(fg, addr)
    end
    values = Vector{Any}(undef, N)
    for addr in elimination_order
        values[addr_to_idx(fg, addr)] = addr_to_value[addr]
    end
    total_log_prob = 0.0
    for addr in reverse(elimination_order)
        fg = intermediate_fgs[addr_to_idx(fg, addr)]
        log_prob = compute_log_prob_addr(fg, values, addr, addr_to_value[addr])
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

#########################################
# compiling a trace into a factor graph #
#########################################

struct AddrInfo{T,U}
    domain::Vector{T}
    parent_addrs::Vector{U}
end

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

function compile_trace_to_factor_graph(trace, info::Dict{Any,AddrInfo})

    # choose order of addresses (note, this is NOT the elimination order)
    # TODO does the order in which the addresses are indexed matter?
    # (it might, somehow relate to elimination ordering?)
    # for now, just choose an arbitrary order?
    addrs = collect(keys(info))
    addr_to_idx = Dict{Any,Int}()
    for (idx, addr) in enumerate(addrs)
        addr_to_idx[addr] = idx
    end

    println("addrs: $addrs")

    # construct factor nodes
    N = length(addrs)
    idx_to_factor_node = Vector{FactorNode{N}}(undef, N)
    factor_id = 1
    for (addr, addr_info) in info

        println("constructing factor for addr $addr and its parents:")

        # iterate over our values, and values of our parents, and populate the factor

        in_factor(a) = (a == addr || a in addr_info.parent_addrs)
        dims = (in_factor(a) ? length(info[a].domain) : 1 for a in addrs)
        println("dims: $dims")
        log_factor = Array{Float64,N}(undef, dims...)

        view_inds = (in_factor(a) ? Colon() : 1 for a in addrs)
        log_factor_view = view(log_factor, view_inds...)
        
        var_addrs = Vector{Any}(undef, length(addr_info.parent_addrs)+1)
        value_idx_lists = Vector{Any}(undef, length(addr_info.parent_addrs)+1)
        i = 1
        for a in addrs
            if in_factor(a)
                var_addrs[i] = a
                value_idx_lists[i] = collect(1:length(info[a].domain))
                i += 1
            end
        end
        @assert i == length(addr_info.parent_addrs)+2

        # populate factor with values by probing trace with update
        # the key idea is that this scales exponentially in maximum number of
        # parents of a variable, not the total number of variables

        for value_idx_tuple in cartesian_product(value_idx_lists)
            println("value_idx_tuple: $value_idx_tuple")
            choices = choicemap()
            for (a, value_idx) in zip(var_addrs, value_idx_tuple)
                choices[a] = info[a].domain[value_idx]
            end
            display(choices)
            (tmp_trace, _, _, _) = update(trace, get_args(trace), map((_)->NoChange(),get_args(trace)), choices)
            # NOTE: semantics of project not exactly aligned -- since project can use any proposal...
            weight = project(tmp_trace, select(addr))
            log_factor_view[value_idx_tuple...] = weight
        end

        factor = exp.(log_factor .- logsumexp(log_factor[:])) # TODO shift the rest of the code to work in log space
        println("factor: factor")
        factor_node = FactorNode(factor_id, Int[addr_to_idx[a] for a in var_addrs], factor)
        factor_id += 1
        idx_to_factor_node[addr_to_idx[addr]] = factor_node
    end

    # compute children and self
    children_and_self = [Set{Int}(i) for i in 1:N]
    for (addr, addr_info) in info
        for parent_addr in addr_info.parent_addrs
            push!(children_and_self[addr_to_idx[parent_addr]], addr_to_idx[addr])
        end
    end

    # construct factor graph
    var_nodes = PersistentHashMap{Int,VarNode}()
    for (addr, addr_info) in info
        factor_nodes = PersistentSet{FactorNode{N}}(
            [idx_to_factor_node[idx] for idx in children_and_self[addr_to_idx[addr]]])
        var_node = VarNode(addr, factor_nodes, addr_info.domain, get_domain_to_idx(addr_info.domain))
        var_nodes = assoc(var_nodes, addr_to_idx[addr], var_node)
    end
    return FactorGraph{N}(length(idx_to_factor_node), var_nodes, addr_to_idx)
end

export AddrInfo, compile_trace_to_factor_graph

###############################
# generative function wrapper #
###############################

struct FactorGraphSamplerTrace <: Gen.Trace
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
    fg = compile_trace_to_factor_graph(model_trace, info)
    (values, log_prob) = sample_and_compute_log_prob(fg, elimination_order)
    choices = choicemap()
    for (addr, value) in values
        choices[addr] = value
    end
    return FactorGraphSamplerTrace(fg, args, choices, log_prob)
end

function Gen.generate(gen_fn::FactorGraphSampler, args::Tuple, choices::ChoiceMap)
    (model_trace, info, elimination_order) = args
    fg = compile_trace_to_factor_graph(model_trace, info)
    values = Dict{Any,Any}()
    for addr in keys(info)
        values[addr] = choices[addr]
    end
    log_prob = compute_log_prob(fg, elimination_order, values)
    trace = FactorGraphSamplerTrace(fg, args, choices, log_prob)
    return (trace, log_prob)
end

Gen.get_args(trace::FactorGraphSamplerTrace) = trace.args
Gen.get_retval(trace::FactorGraphSamplerTrace) = nothing
Gen.get_choices(trace::FactorGraphSamplerTrace) = trace.choices
Gen.get_score(trace::FactorGraphSamplerTrace) = trace.log_prob
Gen.get_gen_fn(trace::FactorGraphSamplerTrace) = compile_and_sample_factor_graph
Gen.project(trace::FactorGraphSamplerTrace, ::EmptyChoiceMap) = 0.0
Gen.has_argument_grads(gen_fn::FactorGraphSampler) = (false, false, false)
Gen.accepts_output_grad(gen_fn::FactorGraphSampler) = false

export compile_and_sample_factor_graph

#########
# tests #
#########

function test_compile_factor_graph()

    @gen function foo()
        x ~ bernoulli(0.6)
        y ~ bernoulli(x ? 0.2 : 0.9)
        z ~ bernoulli((x && y) ? 0.4 : 0.9)
        w ~ bernoulli(z ? 0.4 : 0.5)
    end

    # x 
    # y
    # z
    # w

    # f1: factor for x
    # f2: factor between x and y
    # f3: factor for x, y, z
    # f4: factor for z, w

    trace = simulate(foo, ())
    info = Dict{Any,AddrInfo}()
    info[:x] = AddrInfo([true, false], [])
    info[:y] = AddrInfo([true, false], [:x])
    info[:z] = AddrInfo([true, false], [:x, :y])
    info[:w] = AddrInfo([true, false], [:z])
    fg = compile_trace_to_factor_graph(trace, info)

    @assert fg.num_factors == 4
    @assert length(fg.var_nodes) == 4
    xnode = idx_to_var_node(fg, addr_to_idx(fg, :x))
    ynode = idx_to_var_node(fg, addr_to_idx(fg, :y))
    znode = idx_to_var_node(fg, addr_to_idx(fg, :z))
    wnode = idx_to_var_node(fg, addr_to_idx(fg, :w))
    @assert addr(xnode) == :x
    @assert addr(ynode) == :y
    @assert addr(znode) == :z
    @assert addr(wnode) == :w
    for node in [xnode, ynode, znode, wnode]
        @assert num_values(node) == 2
        @assert idx_to_value(node, 1) == true
        @assert idx_to_value(node, 2) == false
        @assert value_to_idx(node, true) == 1
        @assert value_to_idx(node, false) == 2
    end
    all_factors = Set{FactorNode}()
    for node in [xnode, ynode, znode, wnode]
        union!(all_factors, factor_nodes(node))
    end
    @assert length(all_factors) == 4
    f1 = first(filter((fn) -> (
        length(vars(fn)) == 1 &&
        addr_to_idx(fg, :x) in vars(fn)), all_factors))
    f2 = first(filter((fn) -> (
        length(vars(fn)) == 2 &&
        addr_to_idx(fg, :x) in vars(fn) &&
        addr_to_idx(fg, :y) in vars(fn)), all_factors))
    f3 = first(filter((fn) -> (
        length(vars(fn)) == 3 &&
        addr_to_idx(fg, :x) in vars(fn) &&
        addr_to_idx(fg, :y) in vars(fn) &&
        addr_to_idx(fg, :z) in vars(fn)), all_factors))
    f4 = first(filter((fn) -> (
        length(vars(fn)) == 2 &&
        addr_to_idx(fg, :z) in vars(fn) &&
        addr_to_idx(fg, :w) in vars(fn)), all_factors))

    normed(arr) = arr / sum(arr)
    
    f1_xtrue = factor_value(fg, f1, Dict(addr_to_idx(fg, :x) => true))
    f1_xfalse = factor_value(fg, f1, Dict(addr_to_idx(fg, :x) => false))
    F = [f1_xtrue, f1_xfalse]
    @assert isapprox(normed(F), normed([0.6, 0.4]))

    f2_xtrue_ytrue = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => true))
    f2_xtrue_yfalse = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => false))
    f2_xfalse_ytrue = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => true))
    f2_xfalse_yfalse = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => false))
    F = [f2_xtrue_ytrue, f2_xtrue_yfalse, f2_xfalse_ytrue, f2_xfalse_yfalse]
    @assert isapprox(normed(F), normed([0.2, 0.8, 0.9, 0.1]))

    f3_true_true_true = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => true))
    f3_true_false_true = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => true))
    f3_false_true_true = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => true))
    f3_false_false_true = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => true))
    f3_true_true_false = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => false))
    f3_true_false_false = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => false))
    f3_false_true_false = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => false))
    f3_false_false_false = factor_value(fg, f3, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => false))
    F = [f3_true_true_true, f3_true_false_true, f3_false_true_true, f3_false_false_true, f3_true_true_false, f3_true_false_false, f3_false_true_false, f3_false_false_false]
    @assert isapprox(normed(F), normed([
        0.4, 0.9, 0.9, 0.9,# z true
        0.6, 0.1, 0.1, 0.1# z false
    ]))

    f4_xtrue_ytrue = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => true, addr_to_idx(fg, :w) => true))
    f4_xtrue_yfalse = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => true, addr_to_idx(fg, :w) => false))
    f4_xfalse_ytrue = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => false, addr_to_idx(fg, :w) => true))
    f4_xfalse_yfalse = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => false, addr_to_idx(fg, :w) => false))
    F = [f4_xtrue_ytrue, f4_xtrue_yfalse, f4_xfalse_ytrue, f4_xfalse_yfalse]
    @assert isapprox(normed(F), normed([0.4, 0.6, 0.5, 0.5]))
    
end
test_compile_factor_graph()

function test_eliminate()
end
test_eliminate()

###########
# example #
###########

@gen function foo()
    x ~ bernoulli(0.6)
    y ~ bernoulli(x ? 0.1 : 0.9)
    z ~ bernoulli((x && y) ? 0.4 : 0.9)
    w ~ bernoulli(z ? 0.4 : 0.5)
end

info = Dict{Any,AddrInfo}()
info[:x] = AddrInfo([true, false], [])
info[:y] = AddrInfo([true, false], [:x])
info[:z] = AddrInfo([true, false], [:x, :y])
info[:w] = AddrInfo([true, false], [:z])
# TODO we also need the factors for downstream data that couple them
# but these aren't associated with a single random choice (and the domain of
# that choice wouldn't matter, even if they were)
elimination_order = [:w, :x, :z, :y]

trace = simulate(foo, ())

# test lower level code
#fg = compile_trace_to_factor_graph(trace, info) 
#graphviz = pyimport("graphviz")
#draw_graph(fg, graphviz, "fg1")
#fg = eliminate(fg, :w)
#draw_graph(fg, graphviz, "fg2")
#fg = eliminate(fg, :x)
#draw_graph(fg, graphviz, "fg3")
#fg = eliminate(fg, :z)
#draw_graph(fg, graphviz, "fg4")
#fg = eliminate(fg, :y)
#draw_graph(fg, graphviz, "fg5")

#fg = compile_trace_to_factor_graph(trace, info) 
#println(collect(values(fg.var_nodes)))
#(v, log_prob) = sample_and_compute_log_prob(fg, elimination_order)
#println(v)
#println(log_prob)

# test generative function wrapper
trace, accepted = mh(trace, compile_and_sample_factor_graph, (info, elimination_order))
@assert accepted

end # module
