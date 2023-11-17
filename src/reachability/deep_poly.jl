
@with_kw struct DeepPoly <: Solver
    max_iter::Int64 = 100
end


function init_deep_poly_symbolic_interval(domain)
    # VF = Vector{HalfSpace{Float64, Vector{Float64}}}
    # domain = HPolytope(VF(constraints_list(domain)))

    n = dim(domain)
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n) # for the constant terms
    return SymbolicInterval([I Z], [I Z], domain)
end


struct LayerNegPos{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    W_neg::Matrix{N}
    W_pos::Matrix{N}
    activation::F
end


LayerNegPos(L::Layer) = LayerNegPos(L.weights, L.bias, min.(L.weights, 0),
                                    max.(L.weights, 0), L.activation)
Layer(L::LayerNegPos) = Layer(L.weights, L.bias, L.activation)

n_nodes(L::LayerNegPos) = length(L.bias)
affine_map(L::LayerNegPos, x) = L.weights*x + L.bias


struct NetworkNegPos{N<:Number, V<:Vector{<:LayerNegPos{<:ActivationFunction, N}}} <: AbstractNetwork{N}
    layers::V
end


NetworkNegPos(net::Network) = NetworkNegPos([LayerNegPos(l) for l in net.layers])


function compute_output(nnet::NetworkNegPos, input)
    curr_value = input
    for layer in nnet.layers
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value
end


function interval_map(W_neg::AbstractMatrix{N}, W_pos::AbstractMatrix{N},
                      l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    l_new = W_pos * l .+ W_neg * u
    u_new = W_pos * u .+ W_neg * l

    return (l_new, u_new)
end


function solve(solver::DeepPoly, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("DeepPoly can only handle bounded input sets."))

    nnet, output = problem.network, problem.output
    domain = init_deep_poly_symbolic_interval(problem.input)

    reach = forward_network(solver, nnet, domain, collect=true)
    return reach
end


function forward_linear(solver::DeepPoly, L::LayerNegPos, input::AbstractHyperrectangle)
    # as Hyperrectangle is no subtype of SymbolicInterval, we need separate
    # method for the input layer
    domain = init_deep_poly_symbolic_interval(input)
    return forward_linear(solver, L, domain)
end


function forward_linear(solver::DeepPoly, L::LayerNegPos, input::SymbolicInterval)
    #output_Low, output_Up = interval_map(L.weights, input.Low, input.Up)
    output_Low, output_Up = interval_map(L.W_neg, L.W_pos, input.Low, input.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    return SymbolicInterval(output_Low, output_Up, domain(input))
end


function forward_act(solver::DeepPoly, L::LayerNegPos{ReLU}, input::SymbolicInterval)
    n_node = n_nodes(L)
    output_Low, output_Up = copy(input.Low), copy(input.Up)
    los, his = bounds(input)

    @inbounds for j in 1:n_node
        # up_low, up_up = bounds(upper(input), j)
        # low_low, low_up = bounds(lower(input), j)

        #up_up = upper_bound(upper(input), j)
        #low_low = lower_bound(lower(input), j)
        low_low, up_up = los[j], his[j]

        slope = relaxed_relu_gradient(low_low, up_up)

        output_Up[j, :] .*= slope
        # only -low_low, if ReLU is not fixed
        output_Up[j, end] += slope * max(-low_low, 0)

        if up_up <= 0 || low_low > 0
            output_Low[j, :] .*= slope
        elseif up_up <= abs(low_low)
            output_Low[j, :] .*= 0
        end
        # if up_up > abs(low_low) do nothing, since output_Low is already a
        # copy of input.sym.Low
    end

    return SymbolicInterval(output_Low, output_Up, domain(input))
end


function forward_act(solver::DeepPoly, L::LayerNegPos{Id}, input::SymbolicInterval)
    return input
end


function bounds(sym::SymbolicInterval{<:Hyperrectangle})
    input_set = domain(sym)::Hyperrectangle
    input_los = input_set.center - input_set.radius
    input_his = input_set.center + input_set.radius

    # TODO: maybe use views here instead of copying the subarray into W, b
    W = sym.Low[:, 1:dim(input_set)]
    b = sym.Low[:, end]

    W_pos = max.(W, 0)
    W_neg = min.(W, 0)

    lb = W_pos * input_los .+ W_neg * input_his .+ b

    W = sym.Up[:, 1:dim(input_set)]
    b = sym.Up[:, end]

    # .= because W_pos, W_neg in appropriate size were already allocated before
    W_pos .= max.(W, 0)
    W_neg .= min.(W, 0)

    ub = W_pos * input_his .+ W_neg * input_los .+ b

    return lb, ub
end


# ρ(d, M) is max(bounds(d^T M))
function LazySets.ρ(d::AbstractArray{T,1} where T, sym::NeuralVerification.SymbolicInterval)
    # applying direction to symbolic interval is the same as passing the interval through a linear layer
    # (without bias), the result should be a 1-dimensional
    d_Low, d_Up = NeuralVerification.interval_map(reshape(d, 1, length(d)), sym.Low, sym.Up)
    sym_prime = NeuralVerification.SymbolicInterval(d_Low, d_Up, NeuralVerification.domain(sym))
    up_lo, up_up = bounds(upper(sym_prime), 1)
    return up_up
end
