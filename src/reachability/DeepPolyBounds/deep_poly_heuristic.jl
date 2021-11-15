
@with_kw struct DeepPolyHeuristic <: Solver
    max_vars::Int64 = 10
    var_frac::Float64 = 0.5
end



function forward_linear(solver::DeepPolyHeuristic, L::LayerNegPosIdx, input::SymbolicIntervalHeur)
    output_Low, output_Up = interval_map(L.W_neg, L.W_pos, input.sym.Low, input.sym.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalHeur(sym, input.lbs, input.ubs, input.var_los, input.var_his, input.max_vars, input.importance)
end


function forward_act(solver::DeepPolyHeuristic, L::LayerNegPosIdx{ReLU}, input::SymbolicIntervalHeur)
    n_node = n_nodes(L)
    n_sym = size(input.sym.Low, 2) - 1 # last column is constant term
    n_in = dim(domain(input))
    current_n_vars = n_sym - n_in

    subs_sym_lo, subs_sym_hi = substitute_variables(input.sym.Low, input.sym.Up,
                                                    input.var_los, input.var_his,
                                                    n_in, current_n_vars)
    # TODO: write better function for bounds instead of this monstrosity!!!
    tmp_sym = SymbolicInterval(subs_sym_lo, subs_sym_hi, domain(input))
    los, his = bounds(tmp_sym, input.lbs[L.index], input.ubs[L.index])

    ##### Calculate importance score ######
    crossing = is_crossing.(los, his)
    layer_importance = sum(abs.(subs_sym_lo[crossing, :]), dims=1) .+ sum(abs.(subs_sym_hi[crossing, :]), dims=1)
    importance = input.importance .+ layer_importance[1:end-1]  # constant term doesn't need importance

    n_vars = min(input.max_vars - current_n_vars, floor(Int, solver.var_frac * n_node))
    fv_idxs = fresh_var_largest_range(los, his, n_vars)
    n_vars = length(fv_idxs)

    out_Low, out_Up = zeros(n_node, n_sym + 1 + n_vars), zeros(n_node, n_sym + 1 + n_vars)

    slopes = relaxed_relu_gradient.(los, his)
    out_Up[:, (1:n_sym) ∪ [end]] .= input.sym.Up .* slopes
    out_Up[:, end] .+= slopes .* max.(-los, 0)

    # also apply relaxation to substituted variables
    subs_sym_hi .= subs_sym_hi .* slopes
    subs_sym_hi[:, end] .+= slopes .* max.(-los, 0)

    out_Low[:, (1:n_sym) ∪ [end]] .= input.sym.Low .* relaxed_relu_gradient_lower.(los, his)

    # also apply relaxation to substituted variables
    subs_sym_lo .= subs_sym_lo .* relaxed_relu_gradient_lower.(los, his)

    for (i, v) in enumerate(fv_idxs)
        # store symbolic bounds on fresh variables
        input.var_los[current_n_vars + i, :] .= subs_sym_lo[v, :]
        input.var_his[current_n_vars + i, :] .= subs_sym_hi[v, :]

        # set corresponding entry to unit-vec
        out_Low[v,:] .= unit_vec(n_sym + i, n_sym + 1 + n_vars)
        out_Up[v,:]  .= unit_vec(n_sym + i, n_sym + 1 + n_vars)
    end

    sym = SymbolicInterval(out_Low, out_Up, domain(input))
    output = SymbolicIntervalHeur(sym, input.lbs, input.ubs, input.var_los, input.var_his, input.max_vars, importance)
    output.lbs[L.index] .= los
    output.ubs[L.index] .= his
    return output
end

function forward_act(solver::DeepPolyHeuristic, L::LayerNegPosIdx{Id}, input::SymbolicIntervalHeur)
    return input
end


# TODO: move to some util file
function is_crossing(lb::Float64, ub::Float64)
    lb < 0 && ub > 0 && return true
    return false
end
