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

    # NOTE: when variables get eliminated from a factor graph, they don't get reindex]
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

    # other_values must contain a value for all variables that have a factor in
    # common with variable addr in fg
    var_idx = addr_to_idx(fg, addr)
    var_node = idx_to_var_node(fg, var_idx)
    n = num_values(var_node)
    #println("conditional_dist, addr: $addr, var_idx: $var_idx, num_values: $n")
    probs = ones(n)
    # TODO : writing the slow version first..
    # LATER: use generated function to generate a version that is specialized to N (unroll this loop, and inline the indices..)
    indices = Vector{Int}(undef, N)
    for i in 1:n
        for factor_node in factor_nodes(var_node)
            #println("i: $i, factor_node.id: $(factor_node.id)")
            F::Array{Float64,N} = factor(factor_node)
            fill!(indices, 1)
            for other_var_idx in vars(factor_node)
                #println("other_var_idx: $other_var_idx")
                if other_var_idx != var_idx
                    other_var_node = idx_to_var_node(fg, other_var_idx)
                    indices[other_var_idx] = value_to_idx(other_var_node, values[other_var_idx])
                end
            end
            indices[var_idx] = i
            #println(indices)
            probs[i] = probs[i] * F[CartesianIndex{N}(indices...)]
        end
    end
    return probs / sum(probs)
end

function sample_and_compute_log_prob_addr(fg::FactorGraph, values::Vector{Any}, addr::Any)
    dist = conditional_dist(fg, values, addr)
    idx = categorical(dist)
    value = idx_to_value(idx_to_var_node(fg, addr_to_idx(fg, addr)), idx)
    return (value, log(dist[idx]))
end

function compute_log_prob_addr(fg::FactorGraph, values::Vector{Any}, addr::Any, value::Any)
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
    println("creating factor node for addr: $addr")
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
        println(children_and_self[i])
        factor_nodes = PersistentSet{FactorNode{N}}(
            [addr_to_factor_node[addr] for addr in children_and_self[i]])
        var_node = VarNode(addr, factor_nodes, addr_info.domain, get_domain_to_idx(addr_info.domain))
        var_nodes = assoc(var_nodes, latent_addr_to_idx[addr], var_node)
    end
    return FactorGraph{N}(num_factors, var_nodes, latent_addr_to_idx)
end

export Latent, compile_trace_to_factor_graph

###############################
# generative function wrapper #
###############################

struct FactorGraphSamplerTrace <: Gen.Trace
    fg::FactorGraph
    args::Tuple{Gen.Trace,Dict{Any,Latent},Dict{Any,Observation},Any}
    choices::Gen.DynamicChoiceMap
    log_prob::Float64
end

struct FactorGraphSampler <: GenerativeFunction{Nothing,FactorGraphSamplerTrace}
end

const compile_and_sample_factor_graph = FactorGraphSampler()

function Gen.simulate(gen_fn::FactorGraphSampler, args::Tuple)
    (model_trace, latents, observations, elimination_order) = args
    fg = compile_trace_to_factor_graph(model_trace, latents, observations)
    (values, log_prob) = sample_and_compute_log_prob(fg, elimination_order)
    choices = choicemap()
    for (addr, value) in values
        choices[addr] = value
    end
    return FactorGraphSamplerTrace(fg, args, choices, log_prob)
end

function Gen.generate(gen_fn::FactorGraphSampler, args::Tuple, choices::ChoiceMap)
    (model_trace, latents, observations, elimination_order) = args
    fg = compile_trace_to_factor_graph(model_trace, latents, observations)
    values = Dict{Any,Any}()
    for addr in keys(latents)
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

@gen function foo()
    x ~ bernoulli(0.6)
    y ~ bernoulli(x ? 0.2 : 0.9)
    z ~ bernoulli((x && y) ? 0.4 : 0.9)
    w ~ bernoulli(z ? 0.4 : 0.5)
end

function test_node(fg, addr)
    node = idx_to_var_node(fg, addr_to_idx(fg, addr))
    @assert node.addr == addr
    @assert num_values(node) == 2
    @assert idx_to_value(node, 1) == true
    @assert idx_to_value(node, 2) == false
    @assert value_to_idx(node, true) == 1
    @assert value_to_idx(node, false) == 2
end

normed(arr) = arr / sum(arr)

function test_factor_f1(fg, all_factors)
    f1 = first(filter((fn) -> (
        length(vars(fn)) == 1 &&
        addr_to_idx(fg, :x) in vars(fn)), all_factors))
    
    f1_xtrue = factor_value(fg, f1, Dict(addr_to_idx(fg, :x) => true))
    f1_xfalse = factor_value(fg, f1, Dict(addr_to_idx(fg, :x) => false))
    F = [f1_xtrue, f1_xfalse]
    @assert isapprox(normed(F), normed([0.6, 0.4]))
end

function test_factor_f2(fg, all_factors)
    f2 = first(filter((fn) -> (
        length(vars(fn)) == 2 &&
        addr_to_idx(fg, :x) in vars(fn) &&
        addr_to_idx(fg, :y) in vars(fn)), all_factors))
    f2_xtrue_ytrue = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => true))
    f2_xtrue_yfalse = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => true, addr_to_idx(fg, :y) => false))
    f2_xfalse_ytrue = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => true))
    f2_xfalse_yfalse = factor_value(fg, f2, Dict(addr_to_idx(fg, :x) => false, addr_to_idx(fg, :y) => false))
    F = [f2_xtrue_ytrue, f2_xtrue_yfalse, f2_xfalse_ytrue, f2_xfalse_yfalse]
    @assert isapprox(normed(F), normed([0.2, 0.8, 0.9, 0.1]))
end

function test_factor_f3(fg, all_factors)
    f3 = first(filter((fn) -> (
        length(vars(fn)) == 3 &&
        addr_to_idx(fg, :x) in vars(fn) &&
        addr_to_idx(fg, :y) in vars(fn) &&
        addr_to_idx(fg, :z) in vars(fn)), all_factors))
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
end

function test_factor_f4(fg, all_factors)
    f4 = first(filter((fn) -> (
        length(vars(fn)) == 2 &&
        addr_to_idx(fg, :z) in vars(fn) &&
        addr_to_idx(fg, :w) in vars(fn)), all_factors))
    f4_xtrue_ytrue = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => true, addr_to_idx(fg, :w) => true))
    f4_xtrue_yfalse = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => true, addr_to_idx(fg, :w) => false))
    f4_xfalse_ytrue = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => false, addr_to_idx(fg, :w) => true))
    f4_xfalse_yfalse = factor_value(fg, f4, Dict(addr_to_idx(fg, :z) => false, addr_to_idx(fg, :w) => false))
    F = [f4_xtrue_ytrue, f4_xtrue_yfalse, f4_xfalse_ytrue, f4_xfalse_yfalse]
    @assert isapprox(normed(F), normed([0.4, 0.6, 0.5, 0.5]))
end

function test_factor_f5(fg, all_factors)
    f = first(filter((fn) -> (
        length(vars(fn)) == 1 &&
        addr_to_idx(fg, :z) in vars(fn)), all_factors))
    f_true = factor_value(fg, f, Dict(addr_to_idx(fg, :z) => true))
    f_false = factor_value(fg, f, Dict(addr_to_idx(fg, :z) => false))
    F = [f_true, f_false]
    @assert isapprox(normed(F), normed([0.4 + 0.6, 0.5 + 0.5]))
end

function test_factor_f6(fg, all_factors)
    println("testing factor f6")
    f = first(filter((fn) -> (
        length(vars(fn)) == 2 &&
        addr_to_idx(fg, :y) in vars(fn) &&
        addr_to_idx(fg, :z) in vars(fn)), all_factors))
    f_true_true = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => true))
    f_true_false = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => false))
    f_false_true = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => true))
    f_false_false = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => false))
    F = [f_true_true, f_true_false, f_false_true, f_false_false]
    @assert isapprox(normed(F), normed([
        # x=true         # x=false
        (0.6 * 0.2 * 0.4) + (0.4 * 0.9 * 0.9), # y=true, z=true,
        (0.6 * 0.2 * 0.6) + (0.4 * 0.9 * 0.1),# y=true, z=false,
        (0.6 * 0.8 * 0.9) + (0.4 * 0.1 * 0.9),# y=false, z=true,
        (0.6 * 0.8 * 0.1) + (0.4 * 0.1 * 0.1)# y=false, z=false
    ]))
end

function test_compile_factor_graph()

    # x 
    # y
    # z
    # w

    # f1: factor for x
    # f2: factor between x and y
    # f3: factor for x, y, z
    # f4: factor for z, w

    trace = simulate(foo, ())
    latents = Dict{Any,Latent}()
    latents[:x] = Latent([true, false], [])
    latents[:y] = Latent([true, false], [:x])
    latents[:z] = Latent([true, false], [:x, :y])
    latents[:w] = Latent([true, false], [:z])
    observations = Dict{Any,Observation}()
    fg = compile_trace_to_factor_graph(trace, latents, observations)

    # test nodes
    @assert fg.num_factors == 4
    @assert length(fg.var_nodes) == 4
    for addr in [:x, :y, :z, :w]
        test_node(fg, addr)
    end

    # test factors
    all_factors = Set{FactorNode}()
    for node in values(fg.var_nodes)
        union!(all_factors, factor_nodes(node))
    end
    @assert length(all_factors) == 4
    test_factor_f1(fg, all_factors)
    test_factor_f2(fg, all_factors)
    test_factor_f3(fg, all_factors)
    test_factor_f4(fg, all_factors)

end
test_compile_factor_graph()

function test_eliminate()

    trace = simulate(foo, ())
    latents = Dict{Any,Latent}()
    latents[:x] = Latent([true, false], [])
    latents[:y] = Latent([true, false], [:x])
    latents[:z] = Latent([true, false], [:x, :y])
    latents[:w] = Latent([true, false], [:z])
    observations = Dict{Any,Observation}()
    fg = compile_trace_to_factor_graph(trace, latents, observations)

    # removes factor f4, replaces it with factor f5
    fg = eliminate(fg, :w)

    # test nodes
    @assert fg.num_factors == 5 # (note -- this is the index of the maximum factor)
    @assert length(fg.var_nodes) == 3
    for addr in [:x, :y, :z]
        test_node(fg, addr)
    end

    # test factors
    all_factors = Set{FactorNode}()
    for node in values(fg.var_nodes)
        union!(all_factors, factor_nodes(node))
    end
    @assert length(all_factors) == 4
    test_factor_f1(fg, all_factors)
    test_factor_f2(fg, all_factors)
    test_factor_f3(fg, all_factors)
    println("test factor f5")
    test_factor_f5(fg, all_factors)

    # removes factor f1, f2, and f3, replaces with factor f6 (over y and z)
    fg = eliminate(fg, :x)

    # test nodes
    @assert fg.num_factors == 6 # (note -- this is the index of the maximum factor)
    @assert length(fg.var_nodes) == 2
    for addr in [:y, :z]
        test_node(fg, addr)
    end

    # test factors
    all_factors = Set{FactorNode}()
    for node in values(fg.var_nodes)
        union!(all_factors, factor_nodes(node))
    end
    @assert length(all_factors) == 2
    test_factor_f5(fg, all_factors)
    test_factor_f6(fg, all_factors)
end
test_eliminate()

function test_conditional_dist()

    trace = simulate(foo, ())
    latents = Dict{Any,Latent}()
    latents[:x] = Latent([true, false], [])
    latents[:y] = Latent([true, false], [:x])
    latents[:z] = Latent([true, false], [:x, :y])
    latents[:w] = Latent([true, false], [:z])
    observations = Dict{Any,Observation}()
    fg = compile_trace_to_factor_graph(trace, latents, observations)

    # removes factor f4, replaces it with factor f5
    fg = eliminate(fg, :w)
    # removes factors f1, f2, f3, replace them with factor f6
    fg = eliminate(fg, :x)

    #       x=true              x=false
    F6 = [  (0.6 * 0.2 * 0.4) + (0.4 * 0.9 * 0.9), # y=true, z=true,
            (0.6 * 0.2 * 0.6) + (0.4 * 0.9 * 0.1),# y=true, z=false,
            (0.6 * 0.8 * 0.9) + (0.4 * 0.1 * 0.9),# y=false, z=true,
            (0.6 * 0.8 * 0.1) + (0.4 * 0.1 * 0.1)# y=false, z=false
    ]

    #     z=true     z=false
    F5 = [0.4 + 0.6, 0.5 + 0.5]

    # distribution on z given y
    values = Vector{Any}(undef, 4)
    values[addr_to_idx(fg, :y)] = true
    actual = conditional_dist(fg, values, :z)
    expected = normed([F6[1] * F5[1], F6[2] * F5[2]])
    @show actual
    @show expected
    @assert isapprox(actual, expected)
    values[addr_to_idx(fg, :y)] = false
    actual = conditional_dist(fg, values, :z)
    expected = normed([F6[3] * F5[1], F6[4] * F5[2]])
    @show actual
    @show expected
    @assert isapprox(actual, expected)
end

test_conditional_dist()

function test_mh_accepted()

    trace = simulate(foo, ())
    latents = Dict{Any,Latent}()
    latents[:x] = Latent([true, false], [])
    latents[:y] = Latent([true, false], [:x])
    latents[:z] = Latent([true, false], [:x, :y])
    latents[:w] = Latent([true, false], [:z])
    observations = Dict{Any,Observation}()
    #fg = compile_trace_to_factor_graph(trace, latents, observations)

    # TODO we also need the factors for downstream data that couple them
    # but these aren't associated with a single random choice (and the domain of
    # that choice wouldn't matter, even if they were)
    elimination_order = [:w, :x, :z, :y]
    
    trace = simulate(foo, ())

    for i in 1:100
        println("test: $i")
        # test generative function wrapper
        trace, accepted = mh(trace, compile_and_sample_factor_graph, (latents, observations, elimination_order))
        display(get_choices(trace))
        @assert accepted
    end

end

test_mh_accepted()

###########
# example #
###########


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


end # module
