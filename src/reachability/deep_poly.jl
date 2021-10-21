
@with_kw struct DeepPoly <: Solver
    max_iter::Int64 = 100
end


function init_deep_poly_symbolic_interval(domain)
    VF = Vector{HalfSpace{Float64, Vector{Float64}}}
    domain = HPolytope(VF(constraints_list(domain)))

    n = dim(domain)
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    return SymbolicInterval([I Z], [I Z], domain)
end


function solve(solver::DeepPoly, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("DeepPoly can only handle bounded input sets."))

    nnet, output = problem.network, problem.output
    domain = init_deep_poly_symbolic_interval(problem.input)

    reach = forward_network(solver, nnet, domain, collect=true)
    return reach
end


function forward_linear(solver::DeepPoly, L::Layer, input::SymbolicInterval)
    output_Low, output_Up = interval_map(L.weights, input.Low, input.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    return SymbolicInterval(output_Low, output_Up, domain(input))
end


function forward_act(solver::DeepPoly, L::Layer{ReLU}, input::SymbolicInterval)
    n_node = n_nodes(L)
    output_Low, output_Up = copy(input.Low), copy(input.Up)

    for j in 1:n_node
        up_low, up_up = bounds(upper(input), j)
        low_low, low_up = bounds(lower(input), j)

        slope = relaxed_relu_gradient(low_low, up_up)

        output_Up[j, :] .*= slope
        # only -low_low, if ReLU is not fixed
        output_Up[j, end] += slope * max(-low_low, 0)

        if up_up <= 0 || low_low > 0
            output_Low[j, :] .*= slope
        elseif up_up <= abs(low_low)
            output_Low[j, :] .*= 0
        end
        # if up_up > abs(low_low) do nothing, since output_Low is already a
        # copy of input.sym.Low
    end

    return SymbolicInterval(output_Low, output_Up, domain(input))
end


function forward_act(solver::DeepPoly, L::Layer{Id}, input::SymbolicInterval)
    return input
end
