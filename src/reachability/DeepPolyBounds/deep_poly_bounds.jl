

############### Define the solver: DeepPolyBounds ###############################

struct DeepPolyBounds <: Solver end

###### Forward propagation

function forward_linear(solver::DeepPolyBounds, L::LayerNegPosIdx, input::SymbolicIntervalBounds)
    output_Low, output_Up = interval_map(L.W_neg, L.W_pos, input.sym.Low, input.sym.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalBounds(sym, input.lbs, input.ubs)
end


function relaxed_relu_gradient_lower(l::Real, u::Real)
    u <= 0 && return 0.
    l >= 0 && return 1.
    u <= -l && return 0
    return 1.
end


function forward_act(solver::DeepPolyBounds, L::LayerNegPosIdx{ReLU}, input::SymbolicIntervalBounds)
    n = n_nodes(L)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    los, his = bounds(input.sym, input.lbs[L.index], input.ubs[L.index])

    slopes = relaxed_relu_gradient.(los, his)
    output_Up .*= slopes
    output_Up[:, end] .+= slopes .* max.(-los, 0)

    slopes .= relaxed_relu_gradient_lower.(los, his)
    output_Low .*= slopes

    sym =  SymbolicInterval(output_Low, output_Up, domain(input))
    output = SymbolicIntervalBounds(sym, input.lbs, input.ubs)
    output.lbs[L.index] = los
    output.ubs[L.index] = his
    return output
end


function forward_act(solver::DeepPolyBounds, L::LayerNegPosIdx{Id}, input::SymbolicIntervalBounds)
    return input
end
