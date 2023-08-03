using DifferentialEquations, Optimization
using Zygote, OptimizationOptimJL, OptimizationPolyalgorithms
using Lux, OptimizationOptimisers, OrdinaryDiffEq
using SciMLSensitivity, Random, ComponentArrays
using Statistics: mean

function controller!(
    initial_condition::Vector{Float64},
    p::Vector{Float64},
    h::Float64,
    timeframe::Tuple{Float64,Float64}=(0.0, 30.0),
    maxiters::Int=100;
    loss_step::Int=10,
    rng::AbstractRNG=Random.default_rng()
)
    ann = Lux.Chain(Lux.Dense(6, 16, swish), Lux.Dense(16, 16, swish), Lux.Dense(16, 1, tanh))
    parameters, state = Lux.setup(rng, ann)

    function dudt_(du, u, p, t, p_true)
        η = ann(u, p, state)[1]
        S, E, I, R, D, h = u
        R₀, γ, σ, ω, δ = p_true
        μ = δ / 1111
        du[1] = μ * sum(u) - R₀ * γ * (1 - η[1]) * S * I + ω * R - μ * S # dS
        du[2] = R₀ * γ * (1 - η[1]) * S * I - σ * E - μ * E # dE
        du[3] = σ * E - γ * I - δ * I - μ * I # dI
        du[4] = (1 - δ) * γ * I - ω * R - μ * R # dR
        du[5] = δ * γ * I # dD
        du[6] = -tanh(-du[3] + du[4]) * (1 - η[1]) # dH
    end
    dudt_(du, u, p, t) = dudt_(du, u, p, t, p_true)
    ts = Float32.(collect(0.0:7.0:timeframe[end]))
    ic = initial_condition
    push!(ic, h)
    prob = ODEProblem(dudt_, ic, timeframe, p)

    function predict(p)
        _prob = remake(prob, u0=ic, tspan=timeframe, p=p)
        Array(solve(_prob, Tsit5(), saveat=ts, abstol=1e-10, reltol=1e-10, verbose=false))
    end

    function loss(p)
        pred = predict(p)
        sum(pred[3, :]) + sum(pred[5, :]) / sum(pred[6, :])
    end

    losses = Float64[]
    callback = function (p, l; loss_step=loss_step)
        push!(losses, l)
        if length(losses) % loss_step == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(parameters))

    res1 = Optimization.solve(
        optprob,
        ADAM(),
        callback=callback,
        maxiters=maxiters
    )
    return first(ann(ic, res1.u, state))[1]
end
