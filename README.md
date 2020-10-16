# GenVariableElimination.jl

WARNING: This package is experimental research code.

This package includes several components:

- A procedure for compiling a factor graph from a trace of a [Gen](https://www.gen.dev) generative function

- An implementation of variable elimination for factor graphs

- Generative functions that sample from the exact joint distribution which is obtained via variable elimination in a factor graph

## Compiling a factor graph from a trace of a generative function

Consider the following generative function, which implements a hidden Markov model:

```julia
prior = rand(3); prior = prior / sum(prior)
A = rand(3, 3); A = A ./ sum(A, dims=2)
B = rand(3, 3); B = B ./ sum(B, dims=2)
T = 10

@gen function hmm()
    z = ({(:z, 1)} ~ categorical(prior))
    {(:x, 1)} ~ categorical(B[z,:])
    for t in 2:T
        z = ({(:z, t)} ~ categorical(A[z,:]))
        {(:x, t)} ~ categorical(B[z,:])
    end
end
```

This model contains discrete random variables for each hidden and observed state.
Conditioned on the values of observed random choices, the hidden state random variables have conditional independencies that can be expressed in the following chain-shaped factor graph:

TODO

To generative this factor graph from a trace, we use use the following code.
This code supplies the structure of the factor graph, and the values will be obtained by querying the trace and its generative function for various conditional probabilities of random choices given their parents:

```julia
using GenVariableElimination: Latent, Observation, compile_trace_to_factor_graph

trace = simulate(hmm, ())

latents = Dict{Any,Latent}()
latents[(:z, 1)] = Latent(collect(1:3), [])
for t in 2:T
    latents[(:z, t)] = Latent(collect(1:3), [(:z, t-1)])
end

observations = Dict{Any,Observation}()
for t in 1:T
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

Currently, the user needs to provide the structure of the factor graph, and the system populates the numeric values of potentials (i.e. factors) through many calls to the `update` method in the trace interface, which be slow.
Both of these limitations can be potentially (at least partially) resolved by statically analyzing the model code, instead of relying on a combination of human static analysis and black-box probability queries.

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

## Sampling from the joint distribution of the factor graph

The package also provides generative functions that sample from the exact conditional joint distribution on the latent variables, at the same addresses as in the original source generative function.
This makes it possible to employ this the sampler within the context of other Gen inference code algorithms.

One generative function takes a trace, latents, observations, and elimination order, and internally compiles the factor graph, and runs variable elimination, and then samples from the resulting full joint distribution.
Here is an example of it used within Metropolis-Hastings:
```julia
using GenVariableElimination: compile_and_sample_factor_graph
trace = simulate(hmm, ())
for i in 1:10
    # NOTE: in this case, it is actually unecessary to recompile the factor graph on each iteration
    trace, accepted = mh(trace, compile_and_sample_factor_graph, (latents, observations, elimination_order))
    @test accepted
end
```

A second generative function is provided that takes a precompiled factor graph, and the results of running variable elimination.
This can be much more efficent if you want to sample multiple times from the same joint distribution, as it does not run the (currently somewhat slow) factor graph compilation process each time it is called:
```julia
@gen function my_gen_fn()
    {:ve} ~ sample_factor_graph(factor_graph, elimination_result)
end
```

Both generative functions employ the same algorithm for sampling from the joint distribution, using the result of variable elimination:
The variable elimination result actually contains a sequence of factor graphs generated during variable elimination.
Each random choice is sampled in the reverse order it was eliminated, and the appropriate conditional distribution of each choice given the already sampled choices is computed using the factor graph in the factor graph sequence immediately before the variable was eliminated.

Note that when applied to the HMM, this algorithm, with this particular choice of elimination order, recovers the forward-filtering backwards sampling.

## References

Koller, Daphne, and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.


