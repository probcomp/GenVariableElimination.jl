using Gen
using GenVariableElimination
using PyCall
using ProfileView

###### old code ######

function compute_new_size(log_factors::Vector{Array{Float64,N}}, idx_to_sum_over::Int) where {N}
    sizes = map(size, log_factors)
    tmp_size = Vector{Int}(undef, N)
    for i in 1:N
        tmp_size[i] = maximum(s[i] for s in sizes)
    end
    new_size = copy(tmp_size)
    new_size[idx_to_sum_over] = 1
    return tmp_size, new_size
end

function fast_multiply_and_sum(log_factors::Vector{Array{Float64,N}}, idx_to_sum_over::Int) where {N}

    # first, precompute the output size
    tmp_size, _ = compute_new_size(log_factors, idx_to_sum_over)
    result = zeros(tmp_size...)#Array{Float64, N}(undef, new_size...)
    for log_factor in log_factors
        broadcast!(+, result, log_factor) # BAD
    end

    #result = broadcast(+, log_factors...)
    m = maximum(result, dims=idx_to_sum_over) # BAD
    return m .+ log.(sum(exp.(result .- m), dims=idx_to_sum_over)) # OK
end

####### new code ####

struct Factor{M}
    log_factor::Array{Float64,M}
    var_to_idx::Vector{Int}
    idx_to_var::Vector{Int}
    #vars::NTuple{M,Int}
    #var_to_idx::Dict{Int,Int}
end

function Factor{M}(log_factor, var_to_idx_dict, nvars::Int) where {M}
    var_to_idx = fill(-1, nvars)
    idx_to_var = Vector{Int}(undef, M)
    for (var, idx) in var_to_idx_dict
        var_to_idx[var] = idx
        idx_to_var[idx] = var
    end
    return Factor{M}(log_factor, var_to_idx, idx_to_var)
end

function new_make_prior_log_factor(i::Int, n::Int, num_hidden_states::Int)
    if i == 1
        log_factor = rand(num_hidden_states)
        return Factor{1}(log_factor, Dict{Int,Int}(1 => 1), n)
    else
        log_factor = rand(num_hidden_states, num_hidden_states)
        return Factor{2}(log_factor, Dict{Int,Int}(i-1 => 1, i => 2), n)
    end
end

function new_make_log_factors(n::Int, num_hidden_states::Int)
    log_factors = Vector{Factor}(undef, n)
    for i in 1:n
        log_factors[i] = new_make_prior_log_factor(i, n, num_hidden_states)
    end
    return log_factors
end

function new_fast_multiply_and_sum(log_factors, var_to_sum_over::Int, var_dims::Vector{Int})

    # the set of variables are those that appear in any factor
    # the order in which the variables appear in the factor is arbitrary. we choose one
    var_to_idx = fill(-1, length(var_dims))
    next_idx = 1
    for lf in log_factors
        for var in lf.idx_to_var
            if var_to_idx[var] < 0
                var_to_idx[var] = next_idx
                next_idx += 1
            end
        end
    end
    nvars = next_idx-1
    idx_to_var = Vector{Int}(undef, nvars)
    for (var, idx) in enumerate(var_to_idx)
        if idx > 0
            idx_to_var[idx] = var
        end
    end

    # once the order of the variables is chosen, we can compute the size of the new factor
    new_factor_size = Int[var_dims[var] for var in idx_to_var]

    # then initialize the new factor
    log_factor = Array{Float64,nvars}(undef, new_factor_size...)

    # populate the new factor
    #  loop over entries in the new factor and compute them one by one, by summing
    var_to_sum_dim = var_dims[var_to_sum_over]
    var_vals = Vector{Int}(undef, nvars)
    for i in 1:length(log_factor)
        idx_vals = CartesianIndices(log_factor)[i] # map from the new log factor indices to values
        for var in 1:nvars
            var_vals[var] = idx_vals[idx_to_var[var]] # note: unecessary computation here..
        end
        total = 0.0
        for val in 1:var_to_sum_dim
            prod = 0.0
            for lf_idx in 1:length(log_factors)
                lf = log_factors[lf_idx]
                idx = CartesianIndex((var == var_to_sum_over ? val : var_vals[var] for var in lf.idx_to_var)...)
                prod *= lf.log_factor[idx]
            end
            total += prod
        end
        log_factor[i] = total
    end
    return Factor{nvars}(log_factor, var_to_idx, idx_to_var)
end

### testing code ###

function make_prior_log_factor(i::Int, n::Int, num_hidden_states::Int)
    @assert 1 <= i <= n
    dims = fill(1, n)
    in_factor = fill(false, n)
    if i == 1
        dims[i] = num_hidden_states
        in_factor[i] = true
    else
        dims[i] = num_hidden_states
        dims[i-1] = num_hidden_states
        in_factor[i] = true
        in_factor[i-1] = true
    end
    log_factor = Array{Float64}(undef, dims...)
    view_inds = map(i -> in_factor[i] ? Colon() : 1, 1:n)
    log_factor_view = view(log_factor, view_inds...)
    log_factor_view[:] = log.(1.0 .+ rand(num_hidden_states^(sum(in_factor))))
    return log_factor
end

function make_log_factors(n, num_hidden_states)
    log_factors = Vector{Array{Float64,n}}(undef, n)
    for i in 1:n
        log_factors[i] = make_prior_log_factor(i, n, num_hidden_states)
    end
    return log_factors
end

const nvars = 50
const var_dim = 5

function do_test(all_log_factors, reps)
    for rep in 1:reps
        #GenVariableElimination.multiply_and_sum([all_log_factors[1], all_log_factors[2]], 1)
        fast_multiply_and_sum([all_log_factors[1], all_log_factors[2]], 1)
        #new_fast_multiply_and_sum((all_log_factors[1], all_log_factors[2]), 1, fill(var_dim, nvars))
    end
end

function new_do_test(all_log_factors, reps)
    for rep in 1:reps
        #GenVariableElimination.multiply_and_sum([all_log_factors[1], all_log_factors[2]], 1)
        #fast_multiply_and_sum([all_log_factors[1], all_log_factors[2]], 1)
        new_fast_multiply_and_sum((all_log_factors[1], all_log_factors[2]), 1, fill(var_dim, nvars))
    end
end

#log_factors = make_log_factors(nvars, var_dim)
#do_test(log_factors, 10)
#@time do_test(log_factors, 1000)
#@profview do_test(log_factors, 1000)

log_factors = new_make_log_factors(nvars, var_dim)
new_do_test(log_factors, 10)
@time new_do_test(log_factors, 1000)
@profview new_do_test(log_factors, 1000)

