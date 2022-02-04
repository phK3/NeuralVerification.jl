

function is_crossing(lb::Float64, ub::Float64)
    lb < 0 && ub > 0 && return true
    return false
end


function merge_into_network(network::Network, coeffs::Vector{N} where N<:Number)
    layers = []
    for l in network.layers[1:end-1]
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        σ = l.activation
        push!(layers, Layer(Ŵ, b̂, σ))
    end

    l = network.layers[end]
    if typeof(l.activation) == Id
        Ŵ = coeffs' * l.weights
        b̂ = coeffs' * l.bias
        push!(layers, Layer(Array(Ŵ), [b̂], Id()))
    elseif typeof(l.activation) == ReLU
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        push!(layers, Layer(Ŵ, b̂, ReLU()))
        # just add an additional layer with the linear objective
        push!(layers, Layer(Array(coeffs'), zeros(1), Id()))
    else
        throw(DomainError(l.activation, ": the activation function is not supported!"))
    end

    return Network(layers)
end


# TODO: merge upper function for vectors coeffs into this function
function merge_into_network(network::Network, A::Matrix{N}, b::Vector{N}) where N<:Number
    layers = []
    for l in network.layers[1:end-1]
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        σ = l.activation
        push!(layers, Layer(Ŵ, b̂, σ))
    end

    l = network.layers[end]
    if typeof(l.activation) == Id
        Ŵ = A * l.weights
        # use -b, as we want everything to be satisfied, if output ≤ 0 (Ax ≤ b <-> Ax - b ≤ 0)
        b̂ = A * l.bias - b
        push!(layers, Layer(Ŵ, b̂, Id()))
    elseif typeof(l.activation) == ReLU
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        push!(layers, Layer(Ŵ, b̂, ReLU()))
        # just add an additional layer with the linear objective
        push!(layers, Layer(A, -b, Id()))
    else
        throw(DomainError(l.activation, ": the activation function is not supported!"))
    end

    return Network(layers)
end
