

struct CROWN <: Solver end


struct SymbolicBound
    # B(x) ≤≥ Λx + γ
    Λ::Union{Matrix{Float64}, UniformScaling{Bool}}  # UniformScaling for I
    γ::Union{Vector{Float64}, Float64}  # Float64 for 0.
end


function backward_linear(solver::CROWN, L::Layer, input::SymbolicBound)
    Λ = input.Λ * L.weights
    γ = input.Λ * L.bias .+ input.γ
    return SymbolicBound(Λ, γ)
end


function backward_act(solver::CROWN, L::Layer{ReLU}, input::SymbolicBound, lbs, ubs; upper=false)
    flip = upper ? -1. : 1.  # Λ⁺ and Λ⁻ are flipped for upper bound vs lower bound
    Λ⁺ = max.(flip * input.Λ, 0)
    Λ⁻ = min.(flip * input.Λ, 0)

    λ_l = relaxed_relu_gradient_lower.(lbs, ubs)
    λ_u = relaxed_relu_gradient.(lbs, ubs)

    β_l = zero(lbs)
    β_u = λ_u .* max.(-lbs, 0)

    Λ = flip * (Λ⁻ .* λ_u' + Λ⁺ .* λ_l')
    γ = flip * (Λ⁻ * β_u + Λ⁺ * β_l) .+ input.γ

    return SymbolicBound(Λ, γ)
end


function backward_act(solver::CROWN, L::Layer{Id}, input::SymbolicBound, lbs, ubs; upper=false)
    return input
end


"""
Calculates parameters of linear bounding function of the network via symbolic
backward propagation, given pre-activation bounds of the intermediate neurons.
By default a lower bound is calculated, for upper bound, set upper=true

solver - the solver to use for symbolic backward propagation
net - the network to generate the bounding function for
lbs - lower bounds on the pre-activation values of the intermediate neurons
ubs - upper bounds on the pre-activation values of the intermediate neurons
input - not needed as of now
upper - whether to calculate upper bound, if false, we calculate lower bound
"""
function backward_network(solver, net, lbs, ubs, input; upper=false)
    # assumes that last layer is linear!
    Z = SymbolicBound(I, 0.)
    Z = backward_linear(solver, net.layers[end], Z)
    for i in reverse(1:length(net.layers)-1)
        layer = net.layers[i]

        Ẑ = backward_act(solver, layer, Z, lbs[i], ubs[i], upper=upper)
        Z = backward_linear(solver, layer, Ẑ)
    end

    return Z
end


"""
Calculates pre-activation bounds for each of the neurons in the network starting
at from_layer.

solver - the solver to use for calculating bounds
net - the network the bounds are calculated for
input_set - input constraints for the network (right now LazySet, Hyperrectangle)

from_layer::Int - calculate bounds from that layer onwards to the last layer
lbs - lower bounds to use. Need bounds for all layers from the start to from_layer-1
ubs - upper bounds to use. See lbs for more info
printing - whether to print progress of bounds computation
"""
function calc_bounds(solver::CROWN, net, input_set; from_layer=1, lbs=nothing, ubs=nothing, printing=false)
    layers = from_layer == 1 ? [] : net.layers[1:from_layer-1]
    lbs = isnothing(lbs) ? [] : lbs
    ubs = isnothing(ubs) ? [] : ubs

    for (i, l) in enumerate(net.layers[from_layer:end])
        if printing
            println("Layer ", from_layer + i-1)
        end

        push!(layers, l)
        nn_part = Network(layers)

        Zl = backward_network(solver, nn_part, lbs, ubs, input_set)
        Zu = backward_network(solver, nn_part, lbs, ubs, input_set, upper=true)

        sym = SymbolicInterval([Zl.Λ Zl.γ], [Zu.Λ Zu.γ], input_set)
        lb, ub = bounds(sym)
        push!(lbs, lb)
        push!(ubs, ub)
    end

    return lbs, ubs
end
