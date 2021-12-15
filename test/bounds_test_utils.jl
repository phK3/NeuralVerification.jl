

"""
Samples a point in the hyperrectangle [los, his] uniformly at random.
"""
function sample(los::Vector{Float64}, his::Vector{Float64})
    @assert length(los) == length(his)
    n = length(los)
    return los .+ rand(n) .* (his - los)
end


"""
Computes result and intermediate values of nnet(input).
"""
function compute_values(nnet, input; collect_f=last)
    x = input
    elem = collect_f((x, x))
    collected = [elem]

    for layer in nnet.layers
        x̂ = affine_map(layer, x)
        x = layer.activation(x̂)
        push!(collected, collect_f((x̂, x)))
    end

    return collected
end


"""
Checks, if given bounds [lbs, ubs] on all neurons in nnet are violated by
randomly sampling n points from the input_set.
"""
function check_bounds(nnet, input_set::Hyperrectangle, lbs, ubs; n=100, ϵ_tol=1e-10, print_success=true)
    counter_examples = []
    for i in 1:n
        x = sample(low(input_set), high(input_set))
        values = compute_values(nnet, x, collect_f=first)
        for (i, (lb, ub)) in enumerate(zip(lbs, ubs))
            # lbs, ubs don't count input layer
            lb_correct = all(values[i+1] .>= lb .- ϵ_tol)
            ub_correct = all(values[i+1] .<= ub .+ ϵ_tol)

            if ~lb_correct || ~ub_correct
                max_lb_error = maximum(abs.(min.(0, values[i+1] - lb)))
                max_ub_error = maximum(max.(0, values[i+1] - ub))
                println("layer ", i, " wrong for ", x, ", largest deviations: ", [max_lb_error, max_ub_error])
                push!(counter_examples, x)
            end
        end
    end

    if length(counter_examples) == 0 && print_success
        println("test passed")
    elseif length(counter_examples) > 0
        println("FAIL !!!")
    end

    return counter_examples
end


"""
Checks bounds randomly by generating input set with specified width, calculating
bounds for it according to the generate_bounds function and then sampling from the
input set.
"""
function check_random_bounds(nnet, width, generate_bounds; n=100, n_rand=100, ϵ_tol=1e-10)
    @assert generate_bounds != nothing
    dim = size(nnet.layers[1].weights, 2)
    counter_examples = []

    for i in 1:n_rand
        low = randn(dim)
        high = low .+ width
        input_set = Hyperrectangle(low=low, high=high)

        lbs, ubs = generate_bounds(input_set)

        input_counters = check_bounds(nnet, input_set, lbs, ubs, n=n, ϵ_tol=ϵ_tol)
        if length(input_counters) > 0
            push!(counter_examples, input_counters)
        end
    end

    if length(counter_examples) == 0
        println("All tests passed")
    else
        println("FAIL !!!")
    end

    return counter_examples
end


"""
Creates network with nice weights for hand calculation.
(Nice meaning small integers)

args:
layer_dims (Int64[]) - vector of layer dimensions (including the input layer!)

kwargs:
possible_vals (list) - list to draw weights from, defaults to -1.:1.
"""
function create_nice_random_network(layer_dims::Vector{Int64}; possible_vals=nothing)
    possible_vals = isnothing(possible_vals) ? (-1.:1.) : possible_vals

    layers = []
    n_layers = length(layer_dims) - 1 # because we don't create input layer
    d_last = layer_dims[1]
    for (i, d) in enumerate(layer_dims[2:end])
        weights = rand(possible_vals, d, d_last)
        bias = rand(possible_vals, d)

        if i < n_layers
            push!(layers, NV.Layer(weights, bias, NV.ReLU()))
        else
            push!(layers, NV.Layer(weights, bias, NV.Id()))
        end

        d_last = d
    end

    return Network(layers)
end


"""
Creates network with gaussian distributed weights.

layer_dims (Int64[]) - vector of layer dimensions (including the input layer!)
"""
function create_random_network(layer_dims::Vector{Int64})
    layers = []
    n_layers = length(layer_dims) - 1 # because we don't create input layer
    d_last = layer_dims[1]
    for (i, d) in enumerate(layer_dims[2:end])
        weights = randn(d, d_last)
        bias = randn(d)

        if i < n_layers
            push!(layers, NV.Layer(weights, bias, NV.ReLU()))
        else
            push!(layers, NV.Layer(weights, bias, NV.Id()))
        end

        d_last = d
    end

    return Network(layers)
end
