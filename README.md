# GenVariableElimination.jl

[![Build Status](https://travis-ci.com/probcomp/GenVariableElimination.jl.svg?branch=main)](https://travis-ci.com/probcomp/GenVariableElimination.jl)

WARNING: This package is experimental. The API is not stable, and it has not been not performance-tuned. The implementation has performance issues when the number of variables increases above ~30. (The performance issues are not due to problems with algorithmic complexity, but appear to involve the specific way that factor computations are currently implemented.)

This package includes several components:

- Procedures for compiling factor graphs from [traces](https://www.gen.dev/dev/ref/gfi/#Traces-1) of [Gen.jl](https://www.gen.dev) generative functions.

- An implementation of variable elimination for factor graphs

- Generative functions that sample from the exact joint conditional distribution on selected discrete random choices, using variable elimination in a factor graph

This package can compile factor graphs from traces of generative functions that are constructed using either (i) [Gen Dynamic Modeling Language](https://www.gen.dev/dev/ref/modeling/) (DML),
or using (ii) the [Gen Static Modeling Language](https://www.gen.dev/dev/ref/modeling/#Static-Modeling-Language-1) (SML) together with built-in [control-flow combinators](https://www.gen.dev/dev/ref/combinators/).
When compiling from traces of DML generative functions, the user needs to provide extra information about the dependencies between random choices in the trace, and the domains of the choices.
When compiling from traces of SML + combinators generative function, this information is automatically extracted from the model.

Consider the following DML generative function, which implements a hidden Markov model:
```julia
prior = rand(3); prior = prior / sum(prior)
A = rand(3, 3); A = A ./ sum(A, dims=2)
B = rand(3, 3); B = B ./ sum(B, dims=2)
T = 10

@gen function dml_hmm(T::Int)
    z = ({(:z, 1)} ~ categorical(prior))
    {(:x, 1)} ~ categorical(B[z,:])
    for t in 2:T
        z = ({(:z, t)} ~ categorical(A[z,:]))
        {(:x, t)} ~ categorical(B[z,:])
    end
end
```
This model contains discrete random variables for each hidden and observed state.
Conditioned on the values of observed random choices, the hidden state random variables have conditional independencies that can be expressed in a chain-shaped factor graph.

Consider the following similar generative function constructed with SML and the Unfold combinator:
```julia
@gen (static) function step(t::Int, z_prev::Int)
    z ~ categorical(A[z_prev,:])
    x ~ categorical(B[z,:])
    return z
end

@gen (static) function sml_hmm(T::Int)
    z_init ~ categorical(prior)
    x_init ~ categorical(B[z_init,:])
    steps ~ (Unfold(step))(T-1, z_init)
end
```

We start with the highest-level API, and then subsequent sections describe the internals, using these generative functions as examples.
See `examples/regression.jl` for another example.

## Sampling from the conditional distribution on selected random choices

The package provides functions that generate generative functions that sample from the exact conditional joint distribution on selected latent variables, at the same addresses as in the original source generative function.
This makes it possible to employ this the sampler within the context of other Gen inference code algorithms, like MCMC or SMC.

### Generating samplers specialized to a given trace
The first function returns a generative function that takes no arguments, and samples from the joint conditional distribution on the given set of addresses, which define the set of addresses to sample and the elimination order to use.
There are two variants of this function.
The first applies to SML + combinator traces only, and the second applies to DML traces, but requires the user to provide additional information.
```julia
generate_backwards_sampler_fixed_trace(trace, addresses)
generate_backwards_sampler_fixed_trace(trace, addresses, latents, observations)
```

SML example:
```julia
trace = simulate(sml_hmm, (10,))
addresses = [:z_init, (:steps=>t=>:z for t in 1:9)...]
sampler = generate_backwards_sampler_fixed_trace(trace, addresses)
sampler_trace = simulate(sampler, ())
```

### Generating samplers specialized to the structure of a given trace
The second function only applies to SML + combinator generative functions, and returns a generative function that takes one argument, which is another trace of the model that takes the same control-flow path as the original trace passed at generation time.
This function does analysis of the structure of the trace only once, instead of within the returned generative function.
```julia
generate_backwards_sampler_fixed_structure(trace, addresses)
```

SML example:
```julia
trace = simulate(sml_hmm, (10,))
addresses = [:z_init, (:steps=>t=>:z for t in 1:9)...]
sampler = generate_backwards_sampler_fixed_structure(trace, addresses)
new_trace, _ = mh(trace, select(:z_init))
sampler_trace = simulate(sampler, (new_trace,))
```

### Samplers that take the trace at run-time
Finally, this package also defines two generative functions that take as input the trace and addresses at runtime.

```julia
backwards_sampler_sml(trace, addresses)
backwards_sampler_dml(trace, addresses, latents, observations)
```

SML example:

```julia
trace = simulate(sml_hmm, (10,))

addresses = [:z_init, (:steps=>t=>:z for t in 1:9)...]
for iter in 1:100
    trace, acc = mh(trace, backwards_sampler_sml, (addresses,))
    @assert acc
end
```

**Note**: [Gen.mh](https://www.gen.dev/dev/ref/mcmc/#Gen.metropolis_hastings) when used with a proposal of this form should always accept, since it is equivalent to Gibbs sampling. In some extreme cases where the proposal is nearly deterministic, numerical issues may cause rejections.

DML example:

```julia
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

trace = simulate(dml_hmm, (10,))

addresses = [(:z, t) for t in 1:10]
for iter in 1:100
    trace, acc = mh(trace, backwards_sampler_dml, addresses, latents, observations)
    @assert acc
end
```

All of these generative functions employ the same algorithm for sampling from the joint distribution, using the result of variable elimination:
The variable elimination result actually contains a sequence of factor graphs generated during variable elimination.
Each random choice is sampled in the reverse order it was eliminated, and the appropriate conditional distribution of each choice given the already sampled choices is computed using the factor graph in the factor graph sequence immediately before the variable was eliminated.

Note that when applied to the HMM, this algorithm, with this particular choice of elimination order, recovers the _forward-filtering backwards sampling_ algorithm.

## Compiling a factor graph from a trace of a generative function


To generate the factor graph from the DML trace, we use use the following code.
This code supplies the structure of the factor graph, and the values will be obtained by querying the trace and its generative function for various conditional probabilities of random choices given their parents:
```julia
using GenVariableElimination: Latent, Observation, compile_trace_to_factor_graph

trace = simulate(dml_hmm, (10,))

latents = Dict{Any,Latent}()
latents[(:z, 1)] = Latent(collect(1:3), [])
for t in 2:10
    latents[(:z, t)] = Latent(collect(1:3), [(:z, t-1)])
end

observations = Dict{Any,Observation}()
for t in 1:10
    observations[(:x, t)] = Observation([(:z, t)])
end

factor_graph = compile_trace_to_factor_graph(trace, latents, observations)
```
The keys for the latent and observation dictionaries are the address of the random choices.
Each `Latent` value contains the domain of values taken by the latent random choice (which is assumed to be discrete), and the set of addresses of random choices that are its _parents_ in the Bayesian network represented by the trace. (This notion can be made more precise.)
Each `Observation` value contains the set of addresses of the subset of the latent random choices that are the parents of this observation random variable.
Note that the observation random variable need not be discrete.
Also, note that you only need to define the set of latent variables you want to infer, and only the observation variables that depend on these latent variables--there may be many other random choices in the trace that are not included in either the latents or observation set.

Note that crucially, this technique can be applied to arbitrary generative functions, including generative functions that make use of continuous random choices, and stochastic control flow, etc.

To draw the factor graph, you need a Python environment with the `graphviz` package installed. Run:
```julia
using GenVariableElimination: draw_factor_graph

using PyCall: pyimport
graphviz = pyimport("graphviz")
draw_factor_graph(factor_graph, graphviz, "hmm") # creates file "hmm.pdf"
```

For the SML + Unfold variant of the model, there is a provided analysis that extracts this information automatically:
```julia
addresses = [:z_init, :steps => 1 => :z, :steps => 2 => :z, :steps => 3 => :z]
(_, latents, observations) = factor_graph_analysis(trace, addresses)
```

## Running variable elimination

After constructing the factor graph, run variable elimination in the graph, providing an elimination ordering:
```julia
using GenVariableElimination: variable_elimination

elimination_order = Any[]
for t in 1:T
    push!(elimination_order, (:z, t))
end
elimination_result = variable_elimination(factor_graph, elimination_order)
```
Note that the complexity of variable elimination depends on the elimination order.
Here, we eliminate the hidden states one by one along the chain, starting with the first hidden state.
This results in a computation that closely resembles the _forward algorithm_ for HMMs.

## References

Koller, Daphne, and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.


