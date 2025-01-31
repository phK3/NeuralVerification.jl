abstract type AbstractNetwork{N<:Number} end

"""
    Layer{F, N}

Consists of `weights` and `bias` for linear mapping, and `activation` for nonlinear mapping.
### Fields
 - `weights::Matrix{N}`
 - `bias::Vector{N}`
 - `activation::F`

See also: [`Network`](@ref)
"""
struct Layer{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    activation::F
end

"""
A Vector of layers.

    Network([layer1, layer2, layer3, ...])

See also: [`Layer`](@ref)
"""
struct Network{N<:Number, V<:Vector{<:Layer{<:ActivationFunction, N}}} <: AbstractNetwork{N}
    layers::V # layers includes output layer
end

"""
    n_nodes(L::Layer)

Returns the number of neurons in a layer.
"""
n_nodes(L::Layer) = length(L.bias)




"""
Reads .onnx network from file.

Only fully connected ReLU networks are supported!
"""
function read_onnx_network(network_file; dtype=AbstractFloat)
    ws, bs = load_network(network_file, dtype=dtype)

    # can we just use dtype instead of eltype(eltype(ws)) ?
    layers = Vector{Layer{<:ActivationFunction,eltype(eltype(ws))}}()
    for (W, b) in zip(ws[1:end-1], bs[1:end-1])
        push!(layers, Layer(Float64.(W), b, ReLU()))
    end

    push!(layers, Layer(Float64.(ws[end]), bs[end], Id()))

    return Network(layers)
end
