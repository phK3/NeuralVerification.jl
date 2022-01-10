

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
LazySets.radius(s::SymbolicIntervalFVHeur) = radius(domain(s))
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



function split_symbolic_interval_fv_heur(s::SymbolicIntervalFVHeur{<:Hyperrectangle}, index::Int)
    domain1, domain2 = split(domain(s), index)

    s1 = init_symbolic_interval_fvheur(s, domain1; max_vars=s.max_vars)
    s2 = init_symbolic_interval_fvheur(s, domain2; max_vars=s.max_vars)

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
