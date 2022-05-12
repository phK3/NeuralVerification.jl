

"""
Instead of intrucing fresh variables, DPNeurifyZono relaxes some neurons with a
zonotope.
"""
@with_kw struct DPNeurifyZono <: Solver
    max_zono::Integer = 10
    zono_frac::Float64 = 0.5
    get_zono_idxs = get_fresh_var_idxs_largest_range
end


function forward_linear(solver::DPNeurifyZono, L::LayerNegPosIdx, input::SymbolicIntervalBounds)
    Low, Up = interval_map(L.W_neg, L.W_pos, input.sym.Low, input.sym.Up)
    Low[:, end] .+= L.bias
    Up[:, end] .+= L.bias
    sym = SymbolicInterval(Low, Up, domain(input))
    return SymbolicIntervalBounds(sym, input.lbs, input.ubs)
end


function forward_act(solver::DPNeurifyZono, L::LayerNegPosIdx{ReLU}, input::SymbolicIntervalBounds)
    n = n_nodes(L)
    current_n_vars = size(input.sym.Low, 2) - 1  # TODO: not really n_vars, but n_sym, add field in SymbolicInterval!

    low_low = lower_bounds(input.sym.Low, domain(input), input.lbs[L.index], input.ubs[L.index])
    low_up = upper_bounds(input.sym.Low, domain(input), low_low, input.ubs[L.index])
    up_up = upper_bounds(input.sym.Up, domain(input), low_up, input.ubs[L.index])
    up_low = lower_bounds(input.sym.Up, domain(input), low_low, up_up)

    zono_idxs = solver.get_zono_idxs(solver.max_zono, current_n_vars, low_low, up_up, solver.zono_frac)
    n_zonos = length(zono_idxs)

    λ_l = relaxed_relu_gradient_lower.(low_low, low_up)
    λ_u = relaxed_relu_gradient.(up_low, up_up)
    β_u = -up_low

    n, m = size(input.sym.Low)

    Low = λ_l .* input.sym.Low
    Up  = λ_u .* input.sym.Up
    Up[:, end] .+= λ_u .* max.(β_u, 0)

    G = zeros(n, n_zonos)  # generator matrix for new error terms
    for (i, z) in enumerate(zono_idxs)
        λ = relaxed_relu_gradient(low_low[z], up_up[z])
        β = -low_low[z]
        Low[z,:] = λ * input.sym.Low[z,:]
        Up[z,:] = λ * input.sym.Low[z,:]
        G[z, i] = λ * max(β, 0)  # we can remove the max, as we only introduce zonos for non-fixed ReLUs
    end

    # add all the error terms ϵᵢ ∈ [0, 1] to the input domain
    dom = Hyperrectangle(low=[low(domain(input)); zeros(n_zonos)], high=[high(domain(input)); ones(n_zonos)])
    # add the generator matrix for the error terms to the symbolic bounds
    sym = SymbolicInterval([Low[:, 1:m-1] G Low[:, end]], [Up[:, 1:m-1] G Up[:, end]], dom)

    output = SymbolicIntervalBounds(sym, input.lbs, input.ubs)
    output.lbs[L.index] .= low_low
    output.ubs[L.index] .= up_up
    return output
end


function forward_act(solver::DPNeurifyZono, L::LayerNegPosIdx{Id}, input::SymbolicIntervalBounds)
    low_low = lower_bounds(input.sym.Low, domain(input), input.lbs[L.index], input.ubs[L.index])
    up_up = upper_bounds(input.sym.Up, domain(input), low_low, input.ubs[L.index])
    input.lbs[L.index] .= low_low
    input.ubs[L.index] .= up_up

    return input
end
