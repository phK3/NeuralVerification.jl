

############# Define the solver: Asymmetric symbolic interval propagation #####

struct ESIPSolver <: Solver end


function forward_linear(solver::ESIPSolver, L::Layer, input::AsymESIP)
    equation = L.weights * input.equation
    equation[:, end] += L.bias

    factors = L.weights * input.factors
    return AsymESIP(equation, factors, input.errors, input.domain)
end


function forward_act(solver::ESIPSolver, L::Layer{ReLU}, input::AsymESIP)
    lbs, ubs = bounds_matrix(input)
    crossing_idxs = findall((lbs .< 0) .& (ubs .> 0))

    λ_l = relaxed_relu_gradient_lower.(lbs, ubs)
    λ_u = relaxed_relu_gradient.(lbs, ubs)

    eq = λ_l .* input.equation
    eq_u = λ_u .* input.equation
    eq_u[:, end] .+= λ_u .* max.(-lbs, 0.)

    error = eq_u - eq + (λ_u - λ_l) .* max.(0, input.factors) * input.errors

    factors = λ_l .* input.factors
    factors = [factors partial_I(factors, crossing_idxs)]

    errors = [input.errors; error[crossing_idxs, :]]

    return AsymESIP(eq, factors, errors, input.domain)
end


function forward_act(solver::ESIPSolver, L::Layer{Id}, input::AsymESIP)
    return input
end


"""
Generates matrix [e₁ | e₂ | ... | eₙ] for n ∈ idxs.

params:
A - matrix - determines number of rows of output matrix
idxs - list - determines number of columns and place of ones in output matrix
"""
function partial_I(A, idxs)
    rows, cols = size(A)
    partI = zeros(rows, length(idxs))
    for (i, idx) in enumerate(idxs)
        partI[idx, i] = 1.
    end
    return partI
end
