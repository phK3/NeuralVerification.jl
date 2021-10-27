

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


function forward_act(solver::DeepPolyBounds, L::LayerNegPosIdx{ReLU}, input::SymbolicIntervalBounds)
    n = n_nodes(L)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    los, his = bounds(input.sym, input.lbs[L.index], input.ubs[L.index])

    for j in 1:n
        low, up = los[j], his[j]

        slope = relaxed_relu_gradient(low, up)

        output_Up[j, :] .*= slope
        # only -low_low, if ReLU is not fixed
        output_Up[j, end] += slope * max(-low, 0)

        if up <= 0 || low > 0
            output_Low[j, :] .*= slope
        elseif up <= abs(low)
            output_Low[j, :] .*= 0
        end
        # if up_up > abs(low_low) do nothing, since output_Low is already a
        # copy of input.sym.Low
    end

    sym =  SymbolicInterval(output_Low, output_Up, domain(input))
    output = SymbolicIntervalBounds(sym, input.lbs, input.ubs)
    output.lbs[L.index] = los
    output.ubs[L.index] = his
    return output
end


function forward_act(solver::DeepPolyBounds, L::LayerNegPosIdx{Id}, input::SymbolicIntervalBounds)
    return input
end
