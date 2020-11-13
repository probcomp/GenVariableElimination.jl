using Gen

@dist labeled_cat(labels, probs) = labels[categorical(probs)]

@gen function ve_backwards_sampler(latents, fg, ve_result)
    N = length(latents)
    values = Vector{Any}(undef, N)
    for addr in reverse(ve_result.elimination_order)
        var_idx = addr_to_idx(fg, addr)
        fg = ve_result.intermediate_fgs[var_idx]
        dist = conditional_dist(fg, values, addr)
        value = ({addr} ~ labeled_cat(latents[addr].domain, dist))
        values[var_idx] = value
    end
end

# NOTE: we could stage the computation better, so less is done within the
# generative function at runtime
# NOTE: we could in principle emit static IR instead of DML code

"""
    sampler::GenerativeFunction = generate_backwards_sampler_fixed_trace(trace, addrs)

Generate a generative function that takes no arguments that samples from the conditional distribution on the given addresses.

Only applies to traces of generative functions constructing purely using SML and combinators (currently Map and Unfold).
The addresses must be discrete random choices within finite support and must not affect the control flow in the trace.
The sampler is generated using variable elimination followed by backwards sampling, where the order of the provided addresses defines the elimination order.
The sampler takes no arguments and is specialized to the conditional distribution for the given trace.
"""
function generate_backwards_sampler_fixed_trace(
        trace::Union{Gen.StaticIRTrace,Gen.VectorTrace{Gen.MapType},Gen.VectorTrace{Gen.UnfoldType}}, addrs)
    (_, latents, observations) = factor_graph_analysis(trace, addrs)
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    ve_result = variable_elimination(fg, addrs)
    @gen function sampler()
        {*} ~ ve_backwards_sampler(latents, fg, ve_result)
    end
    return sampler
end

"""
    sampler::GenerativeFunction = generate_backwards_sampler_fixed_trace(
        trace, addrs, latents::Dict{Any,Latent}, observations::Dict{Any,Observation})

Generate a generative function that takes no arguments that samples from the conditional distribution on the given addresses.

The addresses must be discrete random choices within finite support and must not affect the control flow in the trace.
The sampler is generated using variable elimination followed by backwards sampling, where the order of the provided addresses defines the elimination order.
The sampler takes no arguments and is specialized to the conditional distribution for the given trace.
"""
function generate_backwards_sampler_fixed_trace(
        trace, addrs, latents::Dict{Any,Latent}, observations::Dict{Any,Observation})
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    ve_result = variable_elimination(fg, addrs)
    @gen function sampler()
        {*} ~ ve_backwards_sampler(latents, fg, ve_result)
    end
    return sampler
end

"""
    sampler::GenerativeFunction = generate_backwards_sampler(trace, addrs)

Generate a generative function that takes a trace argument that samples from the conditional distribution on the given addresses.

Only applies to traces of generative functions constructing purely using SML and combinators (currently Map and Unfold).
The addresses must be discrete random choices within finite support and must not affect the control flow in the trace.
The sampler is generated using variable elimination followed by backwards sampling, where the order of the provided addresses defines the elimination order.
The sampler is specialized to traces with the same control flow path, but not necessarily the same values, as the trace provided to the generation function.
"""
function generate_backwards_sampler_fixed_structure(
        trace::Union{Gen.StaticIRTrace,Gen.VectorTrace{Gen.MapType},Gen.VectorTrace{Gen.UnfoldType}}, addrs)
    (_, latents, observations) = factor_graph_analysis(trace, addrs)
    @gen function sampler(trace)
        fg = compile_trace_to_factor_graph(trace, latents, observations)
        ve_result = variable_elimination(fg, addrs)
        {*} ~ ve_backwards_sampler(latents, fg, ve_result)
    end
    return sampler
end

@gen function backwards_sampler_dml(trace, addrs, latents::Dict{Any,Latent}, observations::Dict{Any,Observation})
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    ve_result = variable_elimination(fg, addrs)
    {*} ~ ve_backwards_sampler(latents, fg, ve_result)
end

@gen function backwards_sampler_sml(trace, addrs)
    (_, latents, observations) = factor_graph_analysis(trace, addrs)
    fg = compile_trace_to_factor_graph(trace, latents, observations)
    ve_result = variable_elimination(fg, addrs)
    {*} ~ ve_backwards_sampler(latents, fg, ve_result)
end

export generate_backwards_sampler_fixed_structure
export generate_backwards_sampler_fixed_trace
export backwards_sampler_dml, backwards_sampler_sml
