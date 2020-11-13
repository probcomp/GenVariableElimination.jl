using Test
using Gen
using GenVariableElimination

# TODO: these tests poke around at the guts of internal data structures
# the data structures should be refactored and perhaps
# tests should be replaced with higher level tests of the API

conditional_dist = GenVariableElimination.conditional_dist
addr_to_idx = GenVariableElimination.addr_to_idx
idx_to_var_node = GenVariableElimination.idx_to_var_node
idx_to_value = GenVariableElimination.idx_to_value
value_to_idx = GenVariableElimination.value_to_idx
num_values = GenVariableElimination.num_values
FactorNode = GenVariableElimination.FactorNode
factor_nodes = GenVariableElimination.factor_nodes
vars = GenVariableElimination.vars
factor_value = GenVariableElimination.factor_value
eliminate = GenVariableElimination.eliminate


###### static example #####

function compute_next_probs(x1)
    probs = fill(1/20, 20)
    probs[x1] *= 10
    return probs ./ sum(probs)
end


@gen (static) function static_model()
    p1 ~ beta(0.5, 0.5)
    p2 ~ beta(0.5, 0.5)
    p3 ~ beta(0.5, 0.5)
    x1 ~ uniform_discrete(1, 20)
    x2 ~ categorical(compute_next_probs(x1))
    x3 ~ categorical(compute_next_probs(x2))
    x4 ~ categorical(compute_next_probs(x3))
    x5 ~ categorical(compute_next_probs(x4))
    x6 ~ categorical(compute_next_probs(x5))
    x7 ~ bernoulli(x6 > 5 ? p2 : p3)
    x8 ~ bernoulli(x7 ? p2 : p3)
    x9 ~ bernoulli(x8 ? p2 : p3)
    x10 ~ bernoulli(x9 ? p2 : p3)
    x11 ~ bernoulli(x10 ? p2 : p3)
    x12 ~ bernoulli(x11 ? p2 : p3)
    x13 ~ bernoulli(x12 ? p2 : p3)
    x14 ~ bernoulli(x13 ? p2 : p3)
    x15 ~ bernoulli(x14 ? p2 : p3)
    x16 ~ bernoulli(x15 ? p2 : p3)
    x17 ~ bernoulli(x16 ? p2 : p3)
    x18 ~ bernoulli(x17 ? p2 : p3)
    x19 ~ bernoulli(x18 ? p2 : p3)
    x20 ~ bernoulli(x19 ? 0.5 : 0.1)
end

# three hidden states, three observed states
prior = rand(3)
prior = prior / sum(prior)

A = rand(3, 3)
A = A ./ sum(A, dims=2)

B = rand(3, 3)
B = B ./ sum(B, dims=2)

@gen (static) function step(t::Int, z_prev::Int)
    z ~ categorical(A[z_prev,:])
    x ~ categorical(B[z,:])
    return z
end

@gen (static) function hmm(T::Int)
    z_init ~ categorical(prior)
    x_init ~ categorical(B[z_init,:])
    steps ~ (Unfold(step))(T, z_init)
end

@load_generated_functions()

#@testset "static IR basic block" begin
#
    #trace = simulate(static_model, ())
#
    #(ret_ancestors, latents, observations) = GenVariableElimination.forward_analysis(
        #trace, [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8], (), [Set{Any}()])
    #println(ret_ancestors)
    #println(latents)
    #println(observations)
    #
    #sampler = generate_conditional_sampler(trace, [:x1, :x2, :x3])
    #for i in 1:100
        #@time trace, acc = mh(trace, sampler, ())
        #@test acc
    #end
#
#end

@testset "static IR + unfold HMM" begin

    trace = simulate(hmm, (20,))

    (ret_ancestors, latents, observations) = GenVariableElimination.forward_analysis(
        trace, [:z_init, :steps => 1 => :z, :steps => 2 => :z, :steps => 3 => :z], (), [Set{Any}()])
    println(ret_ancestors)
    println(latents)
    println(observations)
    
    sampler = generate_conditional_sampler(trace, [:z_init, (:steps => t => :z for t in 1:20)...])
    for i in 1:100
        @time trace, acc = mh(trace, sampler, ())
        @test acc
    end

end

# end #

@gen function foo()
    x ~ bernoulli(0.6)
    y ~ bernoulli(x ? 0.2 : 0.9)
    z ~ bernoulli((x && y) ? 0.4 : 0.9)
    w ~ bernoulli(z ? 0.4 : 0.5)
end

function test_node(fg, addr)
    node = idx_to_var_node(fg, addr_to_idx(fg, addr))
    @test node.addr == addr
    @test num_values(node) == 2
    @test idx_to_value(node, 1) == true
    @test idx_to_value(node, 2) == false
    @test value_to_idx(node, true) == 1
    @test value_to_idx(node, false) == 2
end

normed(arr) = arr / sum(arr)

function test_factor_f1(fg, all_factors)
    f1 = first(filter((fn) -> (
        length(vars(fn)) == 1 &&
        addr_to_idx(fg, :x) in vars(fn)), all_factors))
    
    f1_xtrue = factor_value(fg, f1, Dict(addr_to_idx(fg, :x) => true))
    f1_xfalse = factor_value(fg, f1, Dict(addr_to_idx(fg, :x) => false))
    F = [f1_xtrue, f1_xfalse]
    @test isapprox(normed(F), normed([0.6, 0.4]))
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
    @test isapprox(normed(F), normed([0.2, 0.8, 0.9, 0.1]))
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
    @test isapprox(normed(F), normed([
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
    @test isapprox(normed(F), normed([0.4, 0.6, 0.5, 0.5]))
end

function test_factor_f5(fg, all_factors)
    f = first(filter((fn) -> (
        length(vars(fn)) == 1 &&
        addr_to_idx(fg, :z) in vars(fn)), all_factors))
    f_true = factor_value(fg, f, Dict(addr_to_idx(fg, :z) => true))
    f_false = factor_value(fg, f, Dict(addr_to_idx(fg, :z) => false))
    F = [f_true, f_false]
    @test isapprox(normed(F), normed([0.4 + 0.6, 0.5 + 0.5]))
end

function test_factor_f6(fg, all_factors)
    f = first(filter((fn) -> (
        length(vars(fn)) == 2 &&
        addr_to_idx(fg, :y) in vars(fn) &&
        addr_to_idx(fg, :z) in vars(fn)), all_factors))
    f_true_true = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => true))
    f_true_false = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => true, addr_to_idx(fg, :z) => false))
    f_false_true = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => true))
    f_false_false = factor_value(fg, f, Dict(addr_to_idx(fg, :y) => false, addr_to_idx(fg, :z) => false))
    F = [f_true_true, f_true_false, f_false_true, f_false_false]
    @test isapprox(normed(F), normed([
        # x=true         # x=false
        (0.6 * 0.2 * 0.4) + (0.4 * 0.9 * 0.9), # y=true, z=true,
        (0.6 * 0.2 * 0.6) + (0.4 * 0.9 * 0.1),# y=true, z=false,
        (0.6 * 0.8 * 0.9) + (0.4 * 0.1 * 0.9),# y=false, z=true,
        (0.6 * 0.8 * 0.1) + (0.4 * 0.1 * 0.1)# y=false, z=false
    ]))
end


@testset "compiling factor graph from trace" begin

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
    @test fg.num_factors == 4
    @test length(fg.var_nodes) == 4
    for addr in [:x, :y, :z, :w]
        test_node(fg, addr)
    end

    # test factors
    all_factors = Set{FactorNode}()
    for node in values(fg.var_nodes)
        union!(all_factors, factor_nodes(node))
    end
    @test length(all_factors) == 4
    test_factor_f1(fg, all_factors)
    test_factor_f2(fg, all_factors)
    test_factor_f3(fg, all_factors)
    test_factor_f4(fg, all_factors)

end

@testset "variable elimination" begin

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
    @test fg.num_factors == 5 # (note -- this is the index of the maximum factor)
    @test length(fg.var_nodes) == 3
    for addr in [:x, :y, :z]
        test_node(fg, addr)
    end

    # test factors
    all_factors = Set{FactorNode}()
    for node in values(fg.var_nodes)
        union!(all_factors, factor_nodes(node))
    end
    @test length(all_factors) == 4
    test_factor_f1(fg, all_factors)
    test_factor_f2(fg, all_factors)
    test_factor_f3(fg, all_factors)
    test_factor_f5(fg, all_factors)

    # removes factor f1, f2, and f3, replaces with factor f6 (over y and z)
    fg = eliminate(fg, :x)

    # test nodes
    @test fg.num_factors == 6 # (note -- this is the index of the maximum factor)
    @test length(fg.var_nodes) == 2
    for addr in [:y, :z]
        test_node(fg, addr)
    end

    # test factors
    all_factors = Set{FactorNode}()
    for node in values(fg.var_nodes)
        union!(all_factors, factor_nodes(node))
    end
    @test length(all_factors) == 2
    test_factor_f5(fg, all_factors)
    test_factor_f6(fg, all_factors)
end

@testset "conditional_dist" begin

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
    @test isapprox(actual, expected)
    values[addr_to_idx(fg, :y)] = false
    actual = conditional_dist(fg, values, :z)
    expected = normed([F6[3] * F5[1], F6[4] * F5[2]])
    @test isapprox(actual, expected)
end

@testset "MH always accepts" begin

    @gen function bar()
        x ~ bernoulli(0.6)
        y ~ bernoulli(x ? 0.2 : 0.9)
        z ~ bernoulli((x && y) ? 0.4 : 0.9)
        w ~ bernoulli(z ? 0.4 : 0.5)
        obs ~ bernoulli((x && w) ? 0.4 : 0.1)
    end

    trace = simulate(foo, ())
    latents = Dict{Any,Latent}()
    latents[:x] = Latent([true, false], [])
    latents[:y] = Latent([true, false], [:x])
    latents[:z] = Latent([true, false], [:x, :y])
    latents[:w] = Latent([true, false], [:z])
    observations = Dict{Any,Observation}()
    observations[:obs] = Observation([:x, :w])

    elimination_order = [:w, :x, :z, :y]
    
    trace = simulate(foo, ())

    function mh_log_acceptance_ratio(trace)
        forward_trace = simulate(compile_and_sample_factor_graph, (trace, latents, observations, elimination_order))
        (new_trace, weight, _, discard) = update(trace, get_args(trace), map((_) -> NoChange(), get_args(trace)), get_choices(forward_trace))
        (backward_trace, _) = generate(compile_and_sample_factor_graph, (trace, latents, observations, elimination_order), discard)
        return weight + get_score(backward_trace) - get_score(forward_trace)
    end

    for i in 1:100
        log_ratio = mh_log_acceptance_ratio(trace)
        @test abs(log_ratio) < 1e-10
    end

end

@testset "HMM forward-filtering backwards sampling" begin

    # three hidden states, three observed states
    prior = rand(3)
    prior = prior / sum(prior)

    A = rand(3, 3)
    A = A ./ sum(A, dims=2)

    B = rand(3, 3)
    B = B ./ sum(B, dims=2)

    T = 10

    @gen function hmm()
        z = ({(:z, 1)} ~ categorical(prior))
        {(:x, 1)} ~ categorical(B[z,:])
        for t in 2:T
            z = ({(:z, t)} ~ categorical(A[z,:]))
            {(:x, t)} ~ categorical(B[z,:])
        end
    end

    latents = Dict{Any,Latent}()
    latents[(:z, 1)] = Latent(collect(1:3), [])
    for t in 2:T
        latents[(:z, t)] = Latent(collect(1:3), [(:z, t-1)])
    end
    observations = Dict{Any,Observation}()
    for t in 1:T
        observations[(:x, t)] = Observation([(:z, t)])
    end

    elimination_order = Any[]
    for t in 1:T
        push!(elimination_order, (:z, t))
    end

    trace = simulate(hmm, ())

    for i in 1:10
        # NOTE: in this case, it is actually unecessary to recompile the factor graph on each iteration
        trace, accepted = mh(trace, compile_and_sample_factor_graph, (latents, observations, elimination_order))
        @test accepted
    end

end


