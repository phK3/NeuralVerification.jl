
############# Network with [W]⁺, [W]⁻ and layers that store their index ########

struct LayerNegPosIdx{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    W_neg::Matrix{N}
    W_pos::Matrix{N}
    activation::F
    index::Int64
end

LayerNegPosIdx(L::Layer, idx::Int64) = LayerNegPosIdx(L.weights, L.bias, min.(L.weights, 0), max.(L.weights, 0),
                                                           L.activation, idx)
n_nodes(L::LayerNegPosIdx) = length(L.bias)
affine_map(L::LayerNegPosIdx, x)  = L.weights*x .+ L.bias


struct NetworkNegPosIdx{N<:Number, V<:Vector{<:LayerNegPosIdx{<:ActivationFunction, N}}} <: AbstractNetwork{N}
    layers::V
end

NetworkNegPosIdx(net::Network) = NetworkNegPosIdx([LayerNegPosIdx(L, i) for (i, L) in enumerate(net.layers)])

function compute_output(nnet::NetworkNegPosIdx, input)
    curr_value = input
    for layer in nnet.layers
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value
end
