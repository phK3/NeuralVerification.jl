

# TODO: make AbstractSymbolicIntervalBounds a subtype of LazySet?
struct SymbolicIntervalFVHeur{F<:AbstractPolytope, N<:Number} <: LazySet{N}
    Low::Matrix{N}
    Up::Matrix{N}
    domain::F
    lbs::Vector{Vector{N}}
    ubs::Vector{Vector{N}}
    var_los::Matrix{N}
    var_his::Matrix{N}
    max_vars::Int
    importance::Vector{N}
    n_layers::Int
end


domain(s::SymbolicIntervalFVHeur) = s.domain
# TODO: should we change this definition? It's not really the radius of the
# symbolic interval, but we need the radius of the domain during splitting
# ALSO: Why do we need LazySets.radius(domain(s)) for NeuralPriorityOptimizer to
# work, why is it not enough to just write radius(domain(s)) ?
LazySets.radius(s::SymbolicIntervalFVHeur) = LazySets.radius(domain(s))
LazySets.dim(s::SymbolicIntervalFVHeur) = size(s.Up, 1)

get_n_sym(s::SymbolicIntervalFVHeur) = size(s.Low, 2) - 1  # number of current symbolic vars
get_n_in(s::SymbolicIntervalFVHeur) = dim(domain(s))  # number of input variables
get_n_vars(s::SymbolicIntervalFVHeur) = get_n_sym(s) - get_n_in(s) # current number of symboli fresh vars


# calculate bounds for a matrix of **equations**, i.e. the lower **or** upper bound of a symbolic interval
function bounds(eq, input_set::H) where H <: Hyperrectangle
    W = eq[:, 1:dim(input_set)]
    b = eq[:, end]

    W⁺ = max.(W, 0)
    W⁻ = min.(W, 0)

    lb = W⁺ * low(input_set) .+ W⁻ * high(input_set) .+ b
    ub = W⁺ * high(input_set) .+ W⁻ * low(input_set) .+ b

    return lb, ub
end


function lower_bounds(eq, input_set::H, lbs, ubs) where H <: Hyperrectangle
    low, up = bounds(eq, input_set)

    low = max.(low, lbs)
    low = min.(low, ubs)

    return low
end


function upper_bounds(eq, input_set::H, lbs, ubs) where H <: Hyperrectangle
    low, up = bounds(eq, input_set)

    up = max.(up, lbs)
    up = min.(up, ubs)

    return up
end


"""
Initialize SymbolicIntervalFVHeur from neural network and input set.
"""
function init_symbolic_interval_fvheur(net::NetworkNegPosIdx, input_set::AbstractPolytope{N}; max_vars=10) where N <: Number
    n = dim(input_set)
    Low = [I zeros(N, n)]
    Up = [I zeros(N, n)]

    lbs = [fill(typemin(N), n_nodes(l)) for l in net.layers]
    ubs = [fill(typemax(N), n_nodes(l)) for l in net.layers]

    var_los = zeros(N, max_vars, dim(input_set) + 1)
    var_his = zeros(N, max_vars, dim(input_set) + 1)

    importance = zeros(N, dim(input_set))

    n_layers = length(net.layers)

    return SymbolicIntervalFVHeur(Low, Up, input_set, lbs, ubs, var_los, var_his,
                                    max_vars, importance, n_layers)
end


"""
Initialize SymbolicIntervalFVHeur from previous SymbolicIntervalFVHeur and new input set.
"""
function init_symbolic_interval_fvheur(s::SymbolicIntervalFVHeur, input_set::AbstractPolytope{N}; max_vars=10) where N <: Number
    n = dim(input_set)
    Low = [I zeros(N, n)]
    Up = [I zeros(N, n)]

    lbs = [copy(ll) for ll in s.lbs]
    ubs = [copy(uu) for uu in s.ubs]

    var_los = zeros(N, max_vars, dim(input_set) + 1)
    var_his = zeros(N, max_vars, dim(input_set) + 1)

    importance = zeros(N, dim(input_set))

    return SymbolicIntervalFVHeur(Low, Up, input_set, lbs, ubs, var_los, var_his,
                                    max_vars, importance, s.n_layers)
end


"""
Initialize SymbolicIntervalFVHeur by changing symbolic lower and upper bound of
existing SymbolicIntervalFVHeur.
"""
function init_symbolic_interval_fvheur(s::SymbolicIntervalFVHeur, Low::Matrix{N}, Up::Matrix{N}) where N <: Number
    return SymbolicIntervalFVHeur(Low, Up, domain(s), s.lbs, s.ubs,
                                    s.var_los, s.var_his, s.max_vars, s.importance, s.n_layers)
end


"""
Directly substitutes all variables in SymbolicIntervalFVHeur s
"""
function substitute_variables(s::SymbolicIntervalFVHeur)
    n_sym = get_n_sym(s)
    n_in = get_n_in(s)
    current_n_vars = get_n_vars(s)
    return substitute_variables(s.Low, s.Up, s.var_los, s.var_his, n_in, current_n_vars)
end


function maximizer(s::SymbolicIntervalFVHeur)
    subs_sym_lo, subs_sym_hi = substitute_variables(s)
    return maximizer(subs_sym_hi, low(domain(s)), high(domain(s)))
end


function minimizer(s::SymbolicIntervalFVHeur)
    subs_sym_lo, subs_sym_hi = substitute_variables(s)
    return minimizer(subs_sym_lo, low(domain(s)), high(domain(s)))
end

function split_symbolic_interval_fv_heur(s::SymbolicIntervalFVHeur{<:Hyperrectangle}, index::Int)
    domain1, domain2 = split(domain(s), index)

    current_n_vars = get_n_vars(s)
    # can't have more vars than parent node?
    s1 = init_symbolic_interval_fvheur(s, domain1; max_vars=current_n_vars)
    s2 = init_symbolic_interval_fvheur(s, domain2; max_vars=current_n_vars)

    return [s1, s2]
end


"""
Splitting based on intermediate importance scores (i.e. coefficients in all crossing ReLUs)
"""
function split_important_interval(s::SymbolicIntervalFVHeur{<:Hyperrectangle})
    radius = high(domain(s)) - low(domain(s))
    # if there are no more crossing ReLUs, importance will be zero vector -> if we
    # multiply it with radius, all inputs are equally important, and argmax always
    # returns the first index -> infinitely loop splitting
    most_important_dim = sum(s.importance) == 0. ? argmax(radius) : argmax(s.importance .* radius)
    return split_symbolic_interval_fv_heur(s, most_important_dim)
end


################################################################################
#### Visualization
################################################################################
# LazySets can plot the reachable sets, if LazySets.σ(a, sym) (returning the
# support vector of the set sym in direction a) is defined.
# While calculating the maximum in direction a is relatively easy, we need to
# encode the problem as an LP to obtain a point inside the set, where that
# maximum is obtained

function LazySets.constraints_list(sym::SymbolicIntervalFVHeur{<:Hyperrectangle})
    n_sym = size(sym.Low, 2) - 1
    n_in = dim(domain(sym))
    current_n_vars = n_sym - n_in

    hList = HalfSpace[]

    hdim = n_sym + dim(sym)  # need variables for all vars in the symbolic bounds as well as for the current neurons

    # constraints for Box inputs
    for i in 1:n_in
        x_current = zeros(hdim)
        x_current[i] = 1.
        push!(hList, HalfSpace(-x_current, -low(sym.domain, i)))
        push!(hList, HalfSpace(x_current, high(sym.domain, i)))
    end

    # constraints for symbolic upper and lower bounds of current neurons (with fresh vars)
    for i in 1:dim(sym)
        # Lower bounds to halfspaces
        x_current = zeros(dim(sym))
        x_current[i] = 1.
        push!(hList, HalfSpace([sym.Low[i, 1:end-1]; -x_current], -sym.Low[i, end]))

        # Upper bounds to halfspaces
        push!(hList, HalfSpace([-sym.Up[i, 1:end-1]; x_current], sym.Up[i, end]))
    end


    # constraints for fresh vars
    for i in 1:current_n_vars
        x_current = zeros(current_n_vars + dim(sym))  # first n_in positions will be held by input variables,
                                                      # then fresh variables, then current neurons.
                                                      # input variables are already in sym.var_los/var_his
        x_current[i] = 1.

        push!(hList, HalfSpace([sym.var_los[i, 1:end-1]; -x_current], -sym.var_los[i, end]))
        push!(hList, HalfSpace([-sym.var_his[i, 1:end-1]; x_current], sym.var_his[i, end]))
    end

    return hList
end


"""
Calculates support vector of symbolic interval with fresh variables by converting it to an HPolytope and solving a an LP.

Only calculation of support vector is expensive, the maximum can be cheaply calculated (see LazySets.ρ(a, sym) implementation below)
"""
function LazySets.σ(a::AbstractVector, sym::SymbolicIntervalFVHeur{<:Hyperrectangle})
    n_sym = size(sym.Low, 2) - 1
    n_in = dim(domain(sym))
    current_n_vars = n_sym - n_in

    # TODO: just use constraints_list

    hList = HalfSpace[]

    hdim = n_sym + dim(sym)  # need variables for all vars in the symbolic bounds as well as for the current neurons

    # constraints for Box inputs
    for i in 1:n_in
        x_current = zeros(hdim)
        x_current[i] = 1.
        push!(hList, HalfSpace(-x_current, -low(sym.domain, i)))
        push!(hList, HalfSpace(x_current, high(sym.domain, i)))
    end

    # constraints for symbolic upper and lower bounds of current neurons (with fresh vars)
    for i in 1:dim(sym)
        # Lower bounds to halfspaces
        x_current = zeros(dim(sym))
        x_current[i] = 1.
        push!(hList, HalfSpace([sym.Low[i, 1:end-1]; -x_current], -sym.Low[i, end]))

        # Upper bounds to halfspaces
        push!(hList, HalfSpace([-sym.Up[i, 1:end-1]; x_current], sym.Up[i, end]))
    end


    # constraints for fresh vars
    for i in 1:current_n_vars
        x_current = zeros(current_n_vars + dim(sym))  # first n_in positions will be held by input variables,
                                                      # then fresh variables, then current neurons.
                                                      # input variables are already in sym.var_los/var_his
        x_current[i] = 1.

        push!(hList, HalfSpace([sym.var_los[i, 1:end-1]; -x_current], -sym.var_los[i, end]))
        push!(hList, HalfSpace([-sym.var_his[i, 1:end-1]; x_current], sym.var_his[i, end]))
    end

    HP = HPolytope(hList)
    # since we aren't interested in values of other variables than the current neurons their influence is zero
    â = [zeros(n_sym); a]
    x̂ = σ(â, HP)

    # variables of current neurons are at the end of the representation
    return x̂[n_sym+1:end]
end


function LazySets.ρ(a::AbstractVector, s::SymbolicIntervalFVHeur{<:Hyperrectangle})
    W⁺ = max.(0, a)'
    W⁻ = min.(0, a)'
    Low, Up = interval_map(W⁻, W⁺, s.Low, s.Up)

    n_sym = size(s.Low, 2) - 1
    n_in = dim(domain(s))
    current_n_vars = n_sym - n_in

    subs_Low, subs_Up = substitute_variables(Low, Up, s.var_los, s.var_his, n_in, current_n_vars)

    up_low, up_up = bounds(subs_Up, domain(s))

    # return a single number, not a vector!
    # up_up should always have a single element anyways
    return up_up[1]
end
