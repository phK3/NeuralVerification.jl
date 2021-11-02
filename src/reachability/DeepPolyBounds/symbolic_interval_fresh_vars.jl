
struct SymbolicIntervalFV{F} <: AbstractSymbolicIntervalBounds{F}
    sym::SymbolicInterval{F}
    lbs::Vector{Vector{Float64}}
    ubs::Vector{Vector{Float64}}
    var_his::Matrix{Float64} # symbolic upper bounds of fresh vars
    var_los::Matrix{Float64} # symbolic lower bounds of fresh vars
end

function init_symbolic_interval_fv(network::NetworkNegPosIdx, input_set; max_vars=10)
    sym = init_deep_poly_symbolic_interval(input_set)
    lbs = [fill(-Inf, n_nodes(l)) for l in network.layers]
    ubs = [fill( Inf, n_nodes(l)) for l in network.layers]
    # symbolic bounds on variables have number of inputs plus constant
    var_his = zeros(max_vars, dim(input_set) + 1)
    var_los = zeros(max_vars, dim(input_set) + 1)
    return SymbolicIntervalFV(sym, lbs, ubs, var_his, var_los)
end

function init_symbolic_interval_fv(s::SymbolicIntervalFV, input_set::AbstractHyperrectangle; max_vars=10)
    sym = init_deep_poly_symbolic_interval(input_set)
    lbs = [copy(lb) for lb in s.lbs]
    ubs = [copy(ub) for ub in s.ubs]
    var_his = zeros(max_vars, dim(input_set) + 1)
    var_los = zeros(max_vars, dim(input_set) + 1)
    return SymbolicIntervalFV(sym, lbs, ubs, var_his, var_los)
end

function substitute_variables(sym_lo, sym_hi, var_los, var_his, n_in, n_vars)
    # for sym_lo
    # should we change the mask to [:, n_in + 1: n_in + n_vars] ???
    var_terms = sym_lo[:, n_in + 1: end - 1]

    var_terms⁺ = max.(var_terms, 0)
    var_terms⁻ = min.(var_terms, 0)

    subs_lb = var_terms⁺ * var_los[1:n_vars, :] .+ var_terms⁻ * var_his[1:n_vars, :] .+ sym_lo[:, (1:n_in) ∪ [end]]

    # for sym_hi
    var_terms .= sym_hi[:, n_in + 1: end - 1]

    var_terms⁺ .= max.(var_terms, 0)
    var_terms⁻ .= min.(var_terms, 0)

    subs_ub = var_terms⁺ * var_his[1:n_vars, :] .+ var_terms⁻ * var_los[1:n_vars, :] .+ sym_hi[:, (1:n_in) ∪ [end]]

    return subs_lb, subs_ub
end

# TODO: only calculate upper bound, don't use bounds which calculates lo and up
function LazySets.ρ(d::AbstractArray{T,1} where T, sym::A where A<:SymbolicIntervalFV)
    n_sym = size(sym.sym.Low, 2) - 1
    n_in = dim(domain(sym))
    current_n_vars = n_sym - n_in

    # substitute before interval_map as symbolic intervals are exact for affine
    # transformations and after substitution, dimension of symlo, symhi is smaller
    subs_sym_lo, subs_sym_hi = substitute_variables(sym.sym.Low, sym.sym.Up, sym.var_los, sym.var_his, n_in, current_n_vars)
    tmp_sym = SymbolicInterval(subs_sym_lo, subs_sym_hi, domain(sym))
    d_Low, d_Up = interval_map(reshape(d, 1, length(d)), tmp_sym.Low, tmp_sym.Up)
    sym_prime = SymbolicInterval(d_Low, d_Up, domain(sym))
    lo, up = bounds(sym_prime)
    # the result should only be one-dimensional
    return up[1]
end

######## Splitting

function split_symbolic_interval_bounds(s::SymbolicIntervalFV{<:Hyperrectangle}, index)
    lbs, ubs = low(domain(s)), high(domain(s))
    split_val = 0.5 * (lbs[index] + ubs[index])

    high1 = copy(ubs)
    high1[index] = split_val
    low2 = copy(lbs)
    low2[index] = split_val

    domain1 = Hyperrectangle(low=lbs, high=high1)
    domain2 = Hyperrectangle(low=low2, high=ubs)

    s1 = init_symbolic_interval_fv(s, domain1)
    s2 = init_symbolic_interval_fv(s, domain2)

    return [s1, s2]
end
