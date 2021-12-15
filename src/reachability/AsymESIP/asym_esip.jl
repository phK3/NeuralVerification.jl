
################ Define Asymmetric Symbolic Error-based Intervals ##############

struct AsymESIP{F<:AbstractPolytope}
    equation
    factors
    errors
    domain::F
end


function init_asym_esip(input::AbstractHyperrectangle)
    n = dim(input)
    equation = [I zeros(n)]
    factors = zeros(n, 0)
    errors = zeros(0, n + 1)  # extra dimension for constants
    return AsymESIP(equation, factors, errors, input)
end


function sym_bounds_matrix(a::AsymESIP)
    eq = a.equation
    sym_lo = eq + min.(0, a.factors) * a.errors
    sym_up = eq + max.(0, a.factors) * a.errors
    return sym_lo, sym_up
end


function bounds_matrix(a::AsymESIP)
    sym_lo, sym_up = sym_bounds_matrix(a)

    los = low(a.domain)
    his = high(a.domain)

    lb = max.(0, sym_lo[:, 1:end-1]) * los + min.(0, sym_lo[:, 1:end-1]) * his + sym_lo[:, end]
    ub = max.(0, sym_up[:, 1:end-1]) * his + min.(0, sym_up[:, 1:end-1]) * los + sym_up[:, end]

    return lb, ub
end
