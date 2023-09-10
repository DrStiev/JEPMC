### -*- Mode: Julia -*-

### Controller.jl
###
### See file LICENSE in top folder for copyright and licensing
### information.

using DifferentialEquations, Optimization, Plots
using Zygote, OptimizationOptimJL, OptimizationPolyalgorithms
using Lux, OptimizationOptimisers, OrdinaryDiffEq
using SciMLSensitivity, Random, ComponentArrays, Enzyme
using Statistics: mean
using DiffEqFlux: swish

"""
    function that implements a NeuralODE controller to learn about the spread of a specific disease and try to mitigate it via non pharmaceutical interventions
    controller(initial_condition::Vector, p_true::Vector = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]; h = rand(), timeframe::Tuple = (0.0, 30.0), maxiters::Int = 100, step = 7.0, loss_step::Int = 10, loss_function = missing, υ_max = 1.0, rng::AbstractRNG = Random.default_rng())

    # Arguments
    - initial_condition::Vector: initial conditions of the problem
    - p_true::Vector: parameters of the disease
    - h::Float64: happiness value associated to the specific problem
    - timeframe::Tuple: timewindow to be analyzed
    - maxiters::Int: maximum number of iterations to be performed by the learning loop of the controller
    - step::Float64: time step of the integration of the ODE system
    - loss_step::Int: number of step after which the callback function print the actual loss value of the loss function
    - loss_function::Function: loss function to be minimized
    - υ_max::Float64: alert threshold. Additional upper limit for the control measurements
    - rng::AbstractRNG: random number generator

    # Returns
    - cumulative value for the countermeasures
"""
function controller(initial_condition::Vector,
    p_true::Vector = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01];
    h = rand(),
    timeframe::Tuple = (0.0, 14.0),
    maxiters::Int = 30,
    step = 5.0,
    patience::Int = 3,
    loss_step::Int = Int(maxiters / 10),
    loss_function = missing,
    doplot::Bool = false,
    id::Int = missing,
    rng::AbstractRNG = Random.default_rng())
    ann = Lux.Chain(Lux.Dense(6, 64, swish),
        Lux.Dense(64, 64, swish),
        Lux.Dense(64, 1, tanh))
    p, state = Lux.setup(rng, ann)

    function dudt_(du, u, p, t)
        S, E, I, R, D, h = u
        R₀, γ, σ, ω, δ, ξ = p_true
        η = abs(ann(u, p, state)[1][1])
        μ = δ / 1111
        du[1] = μ * sum(u[1:5]) - R₀ * γ * (1 - η) * S * I + ω * R - ξ * S - μ * S # dS
        du[2] = R₀ * γ * (1 - η) * S * I - σ * E - μ * E # dE
        du[3] = σ * E - γ * I - δ * I - μ * I # dI
        du[4] = (1 - δ) * γ * I - ω * R + ξ * S - μ * R # dR
        du[5] = δ * γ * I # dD
        du[6] = (1 - η) * (du[1] + du[4]) - (du[2] + du[3] + du[5]) # dH
    end

    ts = collect(0.0:step:timeframe[end])
    ic = vcat(deepcopy(initial_condition), h)
    prob = ODEProblem(dudt_, ic, timeframe, p)

    function predict(p)
        _prob = remake(prob, u0 = ic, tspan = timeframe, p = p)
        Array(solve(_prob,
            Tsit5(),
            saveat = ts,
            abstol = 1e-10,
            reltol = 1e-10,
            verbose = false))
    end

    function l(x)
        loss_function === missing ?
        (sum(abs2, x[2, :]) + sum(abs2, x[3, :]) + sum(abs2, x[5, :])) /
        sum(abs2, x[end, :]) : loss_function
    end

    loss(p) = l(predict(p))

    patience_temp = 0
    losses = Float64[]
    callback = function (p, l; lstep = loss_step)
        push!(losses, l)
        # Exit early if not improving...
        if length(losses) > 1 && l ≥ losses[end - 1]
            patience_temp += 1
            if patience_temp > patience
                return true
            end
        else
            patience_temp = 0
        end

        if length(losses) % lstep == 0
            @debug "Current loss after $(length(losses)) iterations: $(losses[end])"
        end
        return false
    end

    iter = Int(maxiters / 5)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
    res1 = Optimization.solve(optprob,
        ADAM(),
        callback = callback,
        maxiters = maxiters - iter)

    # good to have but very memory consuming
    optprob2 = remake(optprob, u0 = res1.u)
    res2 = Optimization.solve(optprob2,
        Optim.LBFGS(),
        callback = callback,
        maxiters = iter)

    doplot ? display(plot(losses, title = "Loss node $(id)")) : nothing
    return abs((ann(ic, res2.u, state))[1][1])
end

### end of file -- Controller.jl
