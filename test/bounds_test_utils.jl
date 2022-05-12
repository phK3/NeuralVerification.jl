
using NeuralVerification
const NV = NeuralVerification

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
        x̂ = NV.affine_map(layer, x)
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
function check_random_bounds(nnet, width, generate_bounds; n=100, n_rand=100, ϵ_tol=1e-10, print_individual_tests=true)
    @assert generate_bounds != nothing
    dim = size(nnet.layers[1].weights, 2)
    counter_examples = []

    for i in 1:n_rand
        low = randn(dim)
        high = low .+ width
        input_set = Hyperrectangle(low=low, high=high)

        lbs, ubs = generate_bounds(input_set)

        input_counters = check_bounds(nnet, input_set, lbs, ubs, n=n, ϵ_tol=ϵ_tol, print_success=print_individual_tests)
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


function generate_nice_random_input_set(nnet, width; possible_vals=nothing)
    possible_vals = isnothing(possible_vals) ? (-1.:1.) : possible_vals

    dim = size(nnet.layers[1].weights, 2)
    low = rand(possible_vals, dim)
    high = low .+ width
    return Hyperrectangle(low=low, high=high)
end


function generate_counterexample_nn(width, generate_bounds; layer_range=nothing, layer_width_range=nothing,
                                    n_nns=100, n_sample=100, n_in=2, n_out=1)

    layer_range = isnothing(layer_range) ? (1:10) : layer_range
    layer_width_range = isnothing(layer_width_range) ? (2:2) : layer_width_range

    counter_nets = []
    counters = []
    input_sets = []

    for hl in layer_range
        # number of hidden layers with 2 neurons +1 for input layer +1 for output layer
        layers = rand(layer_width_range, hl+2)
        layers[1] = n_in
        layers[end] = n_out

        for n_nn in 1:n_nns
            net = create_nice_random_network(layers)
            input_set = generate_nice_random_input_set(net, 0.1)
            lbs, ubs = generate_bounds(net, input_set)

            counters_current = check_bounds(net, input_set, lbs, ubs, n=n_sample, print_success=false)

            if length(counters_current) > 0
                println("Found counterexample with ", length(layers)-1, " layers!")
                push!(counter_nets, net)
                push!(counters, counters_current)
                push!(input_sets, input_set)
            end
        end
    end

    return counter_nets, counters, input_sets
end
