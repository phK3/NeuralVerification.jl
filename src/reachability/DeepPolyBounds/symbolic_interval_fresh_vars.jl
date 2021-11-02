
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
