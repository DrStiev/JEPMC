using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq
using Plots, CUDA, SciMLSensitivity, Random, ComponentArrays, BSON, OptimizationOptimJL
using ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse, LuxCUDA
import DiffEqFlux: NeuralODE
import Statistics: mean

rng = Xoshiro(1234)
tspan = (0.0f0, 1200.0f0)
ann = Lux.Chain(Lux.Dense(6, 16, swish), Lux.Dense(16, 16, swish), Lux.Dense(16, 1, tanh))
parameters, state = Lux.setup(rng, ann)

# https://medium.com/swlh/neural-ode-for-reinforcement-learning-and-nonlinear-optimal-control-cartpole-problem-revisited-5408018b8d71
h = rand()
function dudt_(du, u, p, t, p_true)
    S, E, I, R, D, h = u
    R₀, γ, σ, ω, δ = p_true
    υ_max = (exp(-4.5 * I) - 1) / (exp(-4.5) - 1)
    η = abs.(ann(u, p, state)[1])[1] ≤ υ_max ? abs.(ann(u, p, state)[1]) : υ_max
    μ = δ / 1111
    du[1] = μ * sum(u) - R₀ * γ * (1 - η[1]) * S * I + ω * R - μ * S # dS
    du[2] = R₀ * γ * (1 - η[1]) * S * I - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I - μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R - μ * R # dR
    du[5] = δ * γ * I # dD
    du[6] = -(du[3] + du[5]) * 3 + (du[4] * (1 - η[1])) # dH
end
dudt_(du, u, p, t) = dudt_(du, u, p, t, p_true)

u0 = [0.999, 0.0, 0.001, 0.0, 0.0, h]
p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
ts = Float32.(collect(0.0:7.0:tspan[end]))
prob = ODEProblem(dudt_, u0, tspan, parameters)
sol = solve(prob, Tsit5())
plot(sol)
abs.(first(ann(u0, res1.u, state)))[1]

function predict(p)
    _prob = remake(prob, u0=u0, tspan=tspan, p=p)
    Array(solve(_prob, Tsit5(), saveat=ts, abstol=1e-10, reltol=1e-10, verbose=false))
end

function loss(p)
    pred = predict(p)
    sum(abs2, pred[3, :]) + sum(abs2, pred[5, :]) / sum(abs2, pred[6, :])
end

losses = Float64[]
callback = function (p, l; iter=10)
    push!(losses, l)
    if length(losses) > 1 && (losses[end-1] - losses[end]) == 0.0
        # early stopping if not improving
        return true
    end
    if length(losses) % iter == 0
        @info "Current loss after $(length(losses)) iterations: $(losses[end])"
        p = plot(solve(remake(prob, p=p), Tsit5()))
        display(p)
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(parameters))

res1 = Optimization.solve(
    optprob,
    ADAM(0.01),
    callback=callback,
    maxiters=100
)
losses
res = abs.(ann(u0, res1.u, state)[1])[1]

optprob2 = remake(optprob, u0=res1.u)
res2 = Optimization.solve(
    optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    maxiters=100
)
res = abs.(ann(u0, res2.u, state)[1])[1]
