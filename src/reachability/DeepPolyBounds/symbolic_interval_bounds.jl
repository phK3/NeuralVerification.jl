

############ Symbolic intervals that store intermediate bounds #################

# must have fields sym, lbs, ubs
abstract type AbstractSymbolicIntervalBounds{F<:AbstractPolytope} end

struct SymbolicIntervalBounds{F} <: AbstractSymbolicIntervalBounds{F}
    sym::SymbolicInterval{F}
    lbs::Vector{Vector{Float64}}
    ubs::Vector{Vector{Float64}}
end

domain(s::A) where A<:AbstractSymbolicIntervalBounds = domain(s.sym)
LazySets.radius(s::A) where A<:AbstractSymbolicIntervalBounds = LazySets.radius(domain(s))

function init_symbolic_interval_bounds(network::NetworkNegPosIdx, input::AbstractHyperrectangle)
    sym = init_deep_poly_symbolic_interval(input)
    lbs = [fill(-Inf, n_nodes(l)) for l in network.layers]
    ubs = [fill( Inf, n_nodes(l)) for l in network.layers]
    return SymbolicIntervalBounds(sym, lbs, ubs)
end

function init_symbolic_interval_bounds(s::SymbolicIntervalBounds, input::AbstractHyperrectangle)
    sym = init_deep_poly_symbolic_interval(input)
    lbs = [copy(lb) for lb in s.lbs]
    ubs = [copy(ub) for ub in s.ubs]
    return SymbolicIntervalBounds(sym, lbs, ubs)
end

function bounds(input::SymbolicInterval, lbs::Vector{Float64}, ubs::Vector{Float64})
    l̂, û = bounds(input)
    l = max.(l̂, lbs)
    u = min.(û, ubs)
    return l, u
end

# TODO: maybe just calculate upper bound instead of both lower and upper bound and then just returning upper bound
function LazySets.ρ(d::AbstractArray{T,1} where T, sym::A where A<:SymbolicIntervalBounds)
    d_Low, d_Up = interval_map(reshape(d, 1, length(d)), sym.sym.Low, sym.sym.Up)
    sym_prime = SymbolicInterval(d_Low, d_Up, domain(sym))
    lo, up = bounds(sym_prime)
    # the result should only be one-dimensional
    return up[1]
end

##### Splitting

function split_symbolic_interval_bounds(s::SymbolicIntervalBounds{<:Hyperrectangle}, index)
    lbs, ubs = low(domain(s)), high(domain(s))
    split_val = 0.5 * (lbs[index] + ubs[index])

    high1 = copy(ubs)
    high1[index] = split_val
    low2 = copy(lbs)
    low2[index] = split_val

    domain1 = Hyperrectangle(low=lbs, high=high1)
    domain2 = Hyperrectangle(low=low2, high=ubs)

    s1 = init_symbolic_interval_bounds(s, domain1)
    s2 = init_symbolic_interval_bounds(s, domain2)

    return [s1, s2]
end

function split_largest_interval(s::SymbolicIntervalBounds{<:Hyperrectangle})
    largest_dimension = argmax(high(domain(s)) - low(domain(s)))
    return split_symbolic_interval_bounds(s, largest_dimension)
end

function split_multiple_times(cell::SymbolicIntervalBounds, n; split=split_largest_interval)
    q = Queue{SymbolicIntervalBounds}()
    enqueue!(q, cell)
    for i in 1:n
        new_cells = split(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end
