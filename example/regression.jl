using Gen
using GenVariableElimination
import Random

function make_data_set(n)
    Random.seed!(1)
    prob_outlier = 0.5
    true_inlier_noise = 0.5
    true_outlier_noise = 5.0
    true_slope = -1
    true_intercept = 2
    xs = collect(range(-5, stop=5, length=n))
    ys = Float64[]
    for (i, x) in enumerate(xs)
        if rand() < prob_outlier
            y = true_slope * x + true_intercept + randn() * true_inlier_noise
        else
            y = true_slope * x + true_intercept + randn() * true_outlier_noise
        end
        push!(ys, y)
    end
    (xs, ys)
end


@gen (static) function datum(x, inlier_std, outlier_std, slope, intercept)
    is_outlier ~ bernoulli(0.5)
    std = is_outlier ? outlier_std : inlier_std
    y ~ normal(x * slope + intercept, std)
    return y
end

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_log_std ~ normal(0, 2)
    outlier_log_std ~ normal(0, 2)
    inlier_std = exp(inlier_log_std)
    outlier_std = exp(outlier_log_std)
    slope ~ normal(0, 2)
    intercept ~ normal(0, 2)
    data ~ Map(datum)(
        xs,
        fill(inlier_std, n),
        fill(outlier_std, n),
        fill(slope, n),
        fill(intercept, n))
end

@gen (static) function slope_proposal(trace)
    slope ~ normal(trace[:slope], 0.5)
end

@gen (static) function intercept_proposal(trace)
    intercept ~ normal(trace[:intercept], 0.5)
end

@gen (static) function inlier_std_proposal(trace)
    inlier_log_std ~ normal(trace[:inlier_log_std], 0.5)
end

@gen (static) function outlier_std_proposal(trace)
    outlier_log_std ~ normal(trace[:outlier_log_std], 0.5)
end

@load_generated_functions()

using PyCall
graphviz = pyimport("graphviz")
function addr_to_name(addr)
    name = replace(replace(replace(replace(string(addr), " => " => "-"), "(" => ""), ")" => ""), ":" => "")
    println("addr: $addr, name: $name")
    return name
end

# for debugging..
function mh_log_acceptance_ratio(trace, proposal, proposal_args)
    forward_trace = simulate(proposal, (trace, proposal_args...))
    println("FORWARD TRACE")
    display(get_choices(forward_trace))
    println(get_score(forward_trace))
    (new_trace, weight, _, discard) = update(trace, get_args(trace), map((_) -> NoChange(), get_args(trace)), get_choices(forward_trace))
    (backward_trace, _) = generate(proposal, (new_trace, proposal_args...))
    return (new_trace, weight, get_score(backward_trace), get_score(forward_trace))
end

function do_inference(xs, ys, num_iters)
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    display(observations)
    println(xs)
    (trace, _) = generate(model, (xs,), observations)
    
    addresses = [:data => i => :is_outlier for i in 1:length(xs)]

    # generate sampler for is_outlier indciator variables
    is_outlier_proposal = generate_backwards_sampler_fixed_structure(trace, addresses)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters

        # steps on the parameters
        for j=1:5
            (trace, _) = metropolis_hastings(trace, slope_proposal, ())
            (trace, _) = metropolis_hastings(trace, intercept_proposal, ())
            (trace, _) = metropolis_hastings(trace, inlier_std_proposal, ())
            (trace, _) = metropolis_hastings(trace, outlier_std_proposal, ())
        end

        # block gibbs step on the is_outlier variables (using variable elimination)
        @time trace, acc = metropolis_hastings(trace, is_outlier_proposal, ())
        println("acc: $acc") # should accept always, except in the beginning where near-deterministic proposals arise

        score = get_score(trace)
        scores[i] = score

        # print
        slope = trace[:slope]
        intercept = trace[:intercept]
        inlier_std = exp(trace[:inlier_log_std])
        outlier_std = exp(trace[:outlier_log_std])
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
    return scores
end

(xs, ys) = make_data_set(30)
do_inference(xs, ys, 10)
@time do_inference(xs, ys, 50)
