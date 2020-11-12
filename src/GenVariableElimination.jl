module GenVariableElimination

include("factor_graph.jl")
include("compiler.jl")
include("gen_fns.jl")

export FactorGraph
export variable_elimination, VariableEliminationResult
export Latent, Observation, compile_trace_to_factor_graph
export draw_factor_graph
export sample_factor_graph, compile_and_sample_factor_graph

# for static IR-
export generate_conditional_sampler

end # module
