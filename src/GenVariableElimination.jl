module GenVariableElimination

include("factor_graph.jl")
include("compiler.jl")
include("gen_fns.jl")

export FactorGraph
export variable_elimination, VariableEliminationResult
export Latent, Observation, compile_trace_to_factor_graph
export draw_factor_graph
export sample_factor_graph, compile_and_sample_factor_graph

export factor_graph_analysis
export generate_backwards_sampler_fixed_trace
export generate_backwards_sampler

# annotate built-in discrete distributions with support information
# (eventually, this will be added to Gen core package)

function is_finite_discrete end
function discrete_finite_support_overapprox end

is_finite_discrete(::Gen.Bernoulli) = true
discrete_finite_support_overapprox(::Gen.Bernoulli, ::Real) = (true, false)

is_finite_discrete(::Gen.Categorical) = true
discrete_finite_support_overapprox(::Gen.Categorical, probs) = (1:length(probs)...,)

is_finite_discrete(::Gen.UniformDiscrete) = true
discrete_finite_support_overapprox(::Gen.UniformDiscrete, low::Integer, high::Integer) = (low:high...,)

export is_finite_discrete
export discrete_finite_support_overapprox

end # module
