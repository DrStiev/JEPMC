### -*- Mode: Julia -*-

### Controller.jl
###
### See file LICENSE in top folder for copyright and licensing
### information.


using DifferentialEquations, Optimization
using Zygote, OptimizationOptimJL, OptimizationPolyalgorithms
using Lux, OptimizationOptimisers, OrdinaryDiffEq
using SciMLSensitivity, Random, ComponentArrays
using Statistics: mean


function controller(initial_condition::Vector{Float64},
                    p_true::Vector{Float64} =
                        [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01],
                    h::Float64 = rand(),
                    timeframe::Tuple{Float64, Float64} = (0.0, 30.0),
                    maxiters::Int = 100;
                    loss_step::Int = 10,
                    υ_max::Float64 = 1.0,
                    rng::AbstractRNG=Random.default_rng()
                    )
    ann = Lux.Chain(Lux.Dense(6, 64, swish),
                    Lux.Dense(64, 64, swish),
                    Lux.Dense(64, 1, tanh))
    p, state = Lux.setup(rng, ann)

    function dudt_(du, u, p, t, p_true)
        S, E, I, R, D, h = u
        R₀, γ, σ, ω, δ, ξ = p_true
        η = abs.(ann(u, p, state)[1])[1]
        η = η ≤ υ_max ? η : υ_max
        μ = δ / 1111
        du[1] = μ * sum(u[1:5]) - R₀ * γ * (1 - η) * S * I + ω * R - ξ * S - μ * S # dS
        du[2] = R₀ * γ * (1 - η) * S * I - σ * E - μ * E # dE
        du[3] = σ * E - γ * I - δ * I - μ * I # dI
        du[4] = (1 - δ) * γ * I - ω * R + ξ * S - μ * R # dR
        du[5] = δ * γ * I # dD
        du[6] = -(du[3] + du[5]) + (du[4] * (1 - η)) # dH
    end
    dudt_(du, u, p, t) = dudt_(du, u, p, t, p_true)
    step = trunc(timeframe[end] / 4.0) # weekly over 30 days
    step = step < 1.0 ? 1.0 : step
    ts = Float32.(collect(0.0:step:timeframe[end]))
    ic = deepcopy(initial_condition)
    push!(ic, h)
    prob = ODEProblem(dudt_, ic, timeframe, p)

    function predict(p)
        _prob = remake(prob, u0 = ic, tspan = timeframe, p = p)
        Array(
            solve(_prob,
                  Tsit5(),
                  saveat = ts,
                  abstol = 1e-10,
                  reltol = 1e-10,
                  verbose = false
                  )
        )
    end

    function loss(p)
        pred = predict(p)
        sum(abs2, pred[3, :]) / sum(abs2, pred[6, :])
    end

    
    losses = Float64[]
    callback = function (p, l; loss_step = loss_step)
        push!(losses, l)
        ## Exit early if not improving...
        if length(losses) > 1 &&
            (abs(losses[end-1] - losses[end])) < eps()
            return true
        end
        if length(losses) % loss_step == 0
            @debug "Current loss after $(length(losses)) iterations: $(losses[end])"
        end
        return false
    end
    
    adtype = Optimization.AutoZygote()
    optf =
        Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob =
        Optimization.OptimizationProblem(optf,
                                         ComponentVector{Float64}(p))
    res1 = Optimization.solve(optprob,
                              ADAM(0.01),
                              callback=callback,
                              maxiters=maxiters
                              )
    return abs.(first(ann(ic, res1.u, state)))[1]
end


### end of file -- Controller.jl
