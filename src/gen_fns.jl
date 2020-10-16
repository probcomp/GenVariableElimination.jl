################################
# generative function wrappers #
################################

# generative function that takes a factor graph and the result of variable
# elimination and samples from the conditional distribution on the latent
# variables

struct SampleFactorGraphTrace <: Gen.Trace
    fg::FactorGraph
    args::Tuple{FactorGraph,VariableEliminationResult}
    choices::Gen.DynamicChoiceMap
    log_prob::Float64
end

struct SampleFactorGraph <: GenerativeFunction{Nothing,SampleFactorGraphTrace}
end

const sample_factor_graph = SampleFactorGraph()

function Gen.simulate(
        ::SampleFactorGraph,
        args::Tuple{FactorGraph{N},VariableEliminationResult}) where {N}
    (fg, elimination_result) = args
    choices = choicemap()
    values = Vector{Any}(undef, N)
    log_prob = 0.0
    for addr in reverse(elimination_result.elimination_order)
        var_idx = addr_to_idx(fg, addr)
        fg = elimination_result.intermediate_fgs[var_idx]
        dist = conditional_dist(fg, values, addr)
        idx = categorical(dist)
        value = idx_to_value(idx_to_var_node(fg, addr_to_idx(fg, addr)), idx)
        values[var_idx] = value
        choices[addr] = value
        log_prob += log(dist[idx])
    end
    return SampleFactorGraphTrace(fg, args, choices, log_prob)
end

function Gen.generate(  
        ::SampleFactorGraph,
        args::Tuple{FactorGraph{N},VariableEliminationResult},
        choices::ChoiceMap) where {N}
    (fg, elimination_result) = args
    values = Vector{Any}(undef, N)
    for addr in elimination_result.elimination_order
        values[addr_to_idx(fg, addr)] = choices[addr]
    end
    log_prob = 0.0
    for addr in reverse(elimination_result.elimination_order)
        fg = elimination_result.intermediate_fgs[addr_to_idx(fg, addr)]
        dist = conditional_dist(fg, values, addr)
        idx = value_to_idx(idx_to_var_node(fg, addr_to_idx(fg, addr)), values[addr_to_idx(fg, addr)])
        log_prob += log(dist[idx])
    end
    trace = SampleFactorGraphTrace(fg, args, DynamicChoiceMap(choices), log_prob)
    return (trace, log_prob)
end

Gen.get_args(trace::SampleFactorGraphTrace) = trace.args
Gen.get_retval(trace::SampleFactorGraphTrace) = nothing
Gen.get_choices(trace::SampleFactorGraphTrace) = trace.choices
Gen.get_score(trace::SampleFactorGraphTrace) = trace.log_prob
Gen.get_gen_fn(trace::SampleFactorGraphTrace) = sample_factor_graph
Gen.project(trace::SampleFactorGraphTrace, ::EmptySelection) = 0.0
Gen.has_argument_grads(::SampleFactorGraph) = (false, false, false)
Gen.accepts_output_grad(::SampleFactorGraph) = false

# TODO: to use this as an MH proposal, it must take its first argument to be a trace
# ideally, we could wrap it in another generatiev function like this:

#@gen function mh_proposal(trace, fg, elimination_result)
    #{*} ~ sample_factor_graph(fg, elimination_result)
#end

# however, this is not supported...
# we could embed it in involutive MH, but unfortunately that requires some code
# in principle, it should be possible to do this without requiring any user code.

# generative function that takes a trace and compiles and samples from the
# conditional distribution on the latent variables

struct CompileAndSampleFactorGraphTrace <: Gen.Trace
    fg::FactorGraph
    args::Tuple{Gen.Trace,Dict{Any,Latent},Dict{Any,Observation},Any}
    choices::Gen.DynamicChoiceMap
    log_prob::Float64
end

struct CompileAndSampleFactorGraph <: GenerativeFunction{Nothing,CompileAndSampleFactorGraphTrace}
end

const compile_and_sample_factor_graph = CompileAndSampleFactorGraph()

function Gen.simulate(
        ::CompileAndSampleFactorGraph,
        args::Tuple{Gen.Trace,Dict{Any,Latent},Dict{Any,Observation},Any})
    (trace, latents, observations, elimination_order) = args
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    elimination_result = variable_elimination(fg, elimination_order)
    trace = Gen.simulate(sample_factor_graph, (fg, elimination_result))
    return CompileAndSampleFactorGraphTrace(fg, args, get_choices(trace), get_score(trace))
end

function Gen.generate(
        ::CompileAndSampleFactorGraph,
        args::Tuple{Gen.Trace,Dict{Any,Latent},Dict{Any,Observation},Any},
        choices::ChoiceMap)
    (trace, latents, observations, elimination_order) = args
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    elimination_result = variable_elimination(fg, elimination_order)
    trace, weight = Gen.generate(sample_factor_graph, (fg, elimination_result), choices)
    return (CompileAndSampleFactorGraphTrace(fg, args, get_choices(trace), get_score(trace)), weight)
end

Gen.get_args(trace::CompileAndSampleFactorGraphTrace) = trace.args
Gen.get_retval(trace::CompileAndSampleFactorGraphTrace) = nothing
Gen.get_choices(trace::CompileAndSampleFactorGraphTrace) = trace.choices
Gen.get_score(trace::CompileAndSampleFactorGraphTrace) = trace.log_prob
Gen.get_gen_fn(trace::CompileAndSampleFactorGraphTrace) = sample_factor_graph
Gen.project(trace::CompileAndSampleFactorGraphTrace, ::EmptySelection) = 0.0
Gen.has_argument_grads(::CompileAndSampleFactorGraph) = (false, false, false)
Gen.accepts_output_grad(::CompileAndSampleFactorGraph) = false
