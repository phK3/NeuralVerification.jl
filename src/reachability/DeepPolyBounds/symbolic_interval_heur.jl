
struct SymbolicIntervalHeur{F} <: AbstractSymbolicIntervalBounds{F}
    sym::SymbolicInterval{F}
    lbs::Vector{Vector{Float64}}
    ubs::Vector{Vector{Float64}}
    var_los::Matrix{Float64} # symbolic lower bounds of fresh vars
    var_his::Matrix{Float64} # symbolic upper bounds of fresh vars
    max_vars::Int64
    importance::Vector{Float64}
end

function init_symbolic_interval_heur(net::NetworkNegPosIdx, input_set; max_vars=10)
    sym = init_deep_poly_symbolic_interval(input_set)
    lbs = [fill(-Inf, n_nodes(l)) for l in net.layers]
    ubs = [fill( Inf, n_nodes(l)) for l in net.layers]

    # symbolic bounds on variables have number of inputs plus constant
    var_his = zeros(max_vars, dim(input_set) + 1)
    var_los = zeros(max_vars, dim(input_set) + 1)

    # importance holds scores for each of the input variables
    importance = zeros(dim(input_set))
    return SymbolicIntervalHeur(sym, lbs, ubs, var_los, var_his, max_vars, importance)
end

function init_symbolic_interval_heur(s::SymbolicIntervalHeur, input_set::AbstractHyperrectangle; max_vars=10)
    sym = init_deep_poly_symbolic_interval(input_set)
    lbs = [copy(lb) for lb in s.lbs]
    ubs = [copy(ub) for ub in s.ubs]
    var_his = zeros(max_vars, dim(input_set) + 1)
    var_los = zeros(max_vars, dim(input_set) + 1)
    importance = zeros(dim(input_set))
    return SymbolicIntervalHeur(sym, lbs, ubs, var_los, var_his, max_vars, importance)
end

function convert2symbolic_interval_fv(s::SymbolicIntervalHeur)
    return SymbolicIntervalFV(s.sym, s.lbs, s.ubs, s.var_los, s.var_his, s.max_vars)
end

# TODO: only calculate upper bound, don't use bounds which calculates lo and up
# TODO: merge ρ for SymbolicIntervalFV and SymbolicIntervalHeur
function LazySets.ρ(d::AbstractArray{T,1} where T, sym::A where A<:SymbolicIntervalHeur)
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


# TODO: factor out into some utils file, don't want to have SymbolicIntervalFV function here
function maximizer(s::Union{SymbolicIntervalHeur, SymbolicIntervalFV})
    n_sym = size(s.sym.Low, 2) - 1
    n_in = dim(domain(s))
    current_n_vars = n_sym - n_in

    subs_sym_lo, subs_sym_hi = substitute_variables(s.sym.Low, s.sym.Up, s.var_los, s.var_his, n_in, current_n_vars)

    return maximizer(subs_sym_hi, low(domain(s)), high(domain(s)))
end

######### Splitting

function split_symbolic_interval_bounds(s::SymbolicIntervalHeur{<:Hyperrectangle}, index::Int64)
    lbs, ubs = low(domain(s)), high(domain(s))
    split_val = 0.5 * (lbs[index] + ubs[index])

    high1 = copy(ubs)
    high1[index] = split_val
    low2 = copy(lbs)
    low2[index] = split_val

    domain1 = Hyperrectangle(low=lbs, high=high1)
    domain2 = Hyperrectangle(low=low2, high=ubs)

    s1 = init_symbolic_interval_heur(s, domain1, max_vars=s.max_vars)
    s2 = init_symbolic_interval_heur(s, domain2, max_vars=s.max_vars)

    return [s1, s2]
end

"""
Splitting based on intermediate importance scores (i.e. coefficients in all crossing ReLUs)
"""
function split_important_interval(s::SymbolicIntervalHeur)
    radius = high(domain(s)) - low(domain(s))
    # if there are no more crossing ReLUs, importance will be zero vector -> if we
    # multiply it with radius, all inputs are equally important, and argmax always
    # returns the first index -> infinitely loop splitting
    most_important_dim = sum(s.importance) == 0. ? argmax(radius) : argmax(s.importance .* radius)
    return split_symbolic_interval_bounds(s, most_important_dim)
end
