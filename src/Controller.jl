### -*- Mode: Julia -*-

### Controller.jl
###
### See file LICENSE in top folder for copyright and licensing
### information.

using DifferentialEquations, Optimization, CUDA, DiffEqGPU
using Zygote, OptimizationOptimJL, OptimizationPolyalgorithms
using Lux, OptimizationOptimisers, OrdinaryDiffEq, LuxCUDA
using SciMLSensitivity, Random, ComponentArrays, Enzyme
using Plots
using DiffEqFlux: swish
using Statistics: mean

@info "GPU device: $(CUDA.device()) functional: $(CUDA.functional())"
@info "LuxCUDA is functional: $(LuxCUDA.functional())"
Lux.gpu_backend!("CUDA")

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
    - rng::AbstractRNG: random number generator

    # Returns
    - cumulative value for the countermeasures
"""
function controller(initial_condition::Vector,
    p_true::Vector = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01, 0.0];
    h = rand(),
    timeframe::Tuple = (0.0, 14.0),
    maxiters = 30,
    step = 5.0,
    patience = 3,
    doplot::Bool = false,
    verbose::Bool = false,
    id = missing,
    rng::AbstractRNG = Random.default_rng())
    CUDA.allowscalar(false)

    ann = Lux.Chain(Lux.Dense(5, 64, swish), Lux.Dense(64, 1))
    p, state = Lux.setup(rng, ann)
    # for reasons gpu_device() is not working properly
    p = p |> ComponentArray |> Lux.cpu_device()
    state = state |> Lux.cpu_device()

    function dudt_(du, u, p, t)
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ, ξ = p_true
        η = abs(ann(u, p, state)[1][1])
        μ = δ / 1111
        du[1] = μ * sum(u[1:5]) - R₀ * γ * (1 - η) * S * I + ω * R - ξ * S - μ * S # dS
        du[2] = R₀ * γ * (1 - η) * S * I - σ * E - μ * E # dE
        du[3] = σ * E - γ * I - δ * I - μ * I # dI
        du[4] = (1 - δ) * γ * I - ω * R + ξ * S - μ * R # dR
        du[5] = δ * γ * I # dD
    end

    ts = Float32.(collect(0.0:step:timeframe[end]))
    initial_condition = Float32.(initial_condition)
    timeframe = Float32.(timeframe)
    prob = ODEProblem(dudt_, initial_condition, timeframe, p)

    function predict(p)
        _prob = remake(prob, u0 = initial_condition, tspan = timeframe, p = p)
        Array(solve(_prob,
            Tsit5(),
            saveat = ts,
            abstol = 1e-10,
            reltol = 1e-10,
            verbose = false)) # suppress unwanted warning. Always active
    end

    function loss(p)
        pred = predict(p)
        (sum(abs2, pred[3, :]) + sum(abs2, pred[5, :]) + sum(abs2, pred[2, :])) / h
    end

    patience_temp = 0
    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        # Exit early if not improving...
        if length(losses) > 1 && (abs(l - losses[end - 1]) < 1e-4 || isinf(l))
            patience_temp += 1
            if patience_temp > patience
                return true
            end
        else
            patience_temp = 0
        end
        return false
    end

    iter = 0 # Int(maxiters / 5)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    res1 = Optimization.solve(optprob,
        ADAM(0.001),
        callback = callback,
        maxiters = maxiters - iter)

    # good to have but very memory consuming
    # optprob2 = remake(optprob, u0 = res1.u)
    # res2 = Optimization.solve(optprob2,
    #     Optim.LBFGS(initial_stepnorm = 0.01),
    #     callback = callback,
    #     maxiters = iter)

    res = abs((ann(initial_condition, res1.u, state))[1][1])
    # res = abs((ann(initial_condition, res2.u, state))[1][1])
    # sometimes the value skyrockets so it needs to be capped
    res -= floor(res)
    doplot ? display(plot(losses, title = "Loss node $(id)")) : nothing
    verbose ?
    println("Current loss after $(length(losses)) iterations: $(losses[end]) \nCountermeasure value for agent $(id) is $(res)") :
    nothing
    return Float64(res)
end

### end of file -- Controller.jl
