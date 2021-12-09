

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
function check_bounds(nnet, input_set::Hyperrectangle, lbs, ubs; n=100)
    counter_examples = []
    for i in 1:n
        x = sample(low(input_set), high(input_set))
        values = compute_values(nnet, x, collect_f=first)
        for (i, (lb, ub)) in enumerate(zip(lbs, ubs))
            # lbs, ubs don't count input layer
            lb_correct = all(values[i+1] .>= lb)
            ub_correct = all(values[i+1] .<= ub)

            if ~lb_correct || ~ub_correct
                println("layer ", i, " wrong for ", x)
                push!(counter_examples, x)
            end
        end
    end

    if length(counter_examples) == 0
        println("test passed")
    else
        println("FAIL !!!")
    end

    return counter_examples
end


"""
Checks bounds randomly by generating input set with specified width, calculating
bounds for it according to the generate_bounds function and then sampling from the
input set.
"""
function check_random_bounds(nnet, width, generate_bounds; n=100, n_rand=100)
    @assert generate_bounds != nothing
    dim = size(nnet.layers[1].weights, 2)
    counter_examples = []

    for i in 1:n_rand
        low = randn(dim)
        high = low .+ width
        input_set = Hyperrectangle(low=low, high=high)

        lbs, ubs = generate_bounds(input_set)

        input_counters = check_bounds(nnet, input_set, lbs, ubs, n=n)
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
