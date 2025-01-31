

##################### Define DeepPolyFreshVars Solver ##########################

@with_kw struct DeepPolyFreshVars <: Solver
    max_vars::Int64 = 10 # maximum number of fresh variables to introduce
    var_frac::Float64 = 0.5 # at most var_frac * n_neurons are introduced per layer
end


######################## Forward Propagation ###################################


function forward_linear(solver::DeepPolyFreshVars, L::LayerNegPosIdx, input::SymbolicIntervalFV)
    output_Low, output_Up = interval_map(L.W_neg, L.W_pos, input.sym.Low, input.sym.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalFV(sym, input.lbs, input.ubs, input.var_los, input.var_his, input.max_vars)
end


function forward_act(solver::DeepPolyFreshVars, L::LayerNegPosIdx{ReLU}, input::SymbolicIntervalFV)
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
    output = SymbolicIntervalFV(sym, input.lbs, input.ubs, input.var_los, input.var_his, input.max_vars)
    output.lbs[L.index] .= los
    output.ubs[L.index] .= his
    return output
end


function forward_act(solver::DeepPolyFreshVars, L::LayerNegPosIdx{Id}, input::SymbolicIntervalFV)
    return input
end


########################### Helper Functions ###################################


"""
Returns n indices of entries in lbs, ubs with largest range (ubs[i] - lbs[i])
"""
function fresh_var_largest_range(lbs::Vector{Float64}, ubs::Vector{Float64}, n::Int64)
    n <= 0 && return Int64[]  # early termination, if no variables are introduced

    ranges = ubs - lbs
    p = sortperm(-ranges) # sort descending
    p = p[(lbs[p] .< 0) .& (ubs[p] .> 0)] # want only crossing ReLUs
    return p[1:min(length(p), n)]
end

# TODO: remove after debugging!!!
function fresh_var_first(lbs::Vector{Float64}, ubs::Vector{Float64}, n::Int64)
    crossing_mask = ((lbs .< 0) .& (ubs .> 0))
    crossing = (1:length(lbs))[crossing_mask]
    return crossing[1:min(length(crossing), n)]
end


function unit_vec(i, n)
    e_i = zeros(n)
    e_i[i] = 1.
    return e_i
end
