# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/
# TODO: https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/
# TODO: https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL
using DifferentialEquations, Lux, SciMLSensitivity, ComponentArrays, OptimizationOptimisers
using Distributions, Random, Plots

function F!(du, u, p, t)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ, η, ξ = p
    du[1] = -R₀ * γ * (1 - η) * S * I + ω * R - ξ * S # dS
    du[2] = R₀ * γ * (1 - η) * S * I - σ * E # dE
    du[3] = σ * E - γ * I - δ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R + ξ * S # dR
    du[5] = δ * γ * I # dD
end

condition_voc(u, t, integrator) = rand(rng) < 8e-3

function affect_voc!(integrator)
    println("voc")
    integrator.p[1] = rand(rng, Uniform(3.3, 5.7))
    integrator.p[2] = abs(rand(rng, Normal(integrator.p[2], integrator.p[2] / 10)))
    integrator.p[3] = abs(rand(rng, Normal(integrator.p[3], integrator.p[3] / 10)))
    integrator.p[4] = abs(rand(rng, Normal(integrator.p[4], integrator.p[4] / 10)))
    integrator.p[5] = abs(rand(rng, Normal(integrator.p[5], integrator.p[5] / 10)))
end

voc_cb = ContinuousCallback(condition_voc, affect_voc!)
cb = CallbackSet(voc_cb)

rng = Xoshiro(1234)
u = [(1e6 - 1) / 1e6, 0, 1 / 1e6, 0, 0]
datasize = 171 # ≈ 1 entrata a settimana
tspan = (0.0f0, 1200.0f0)
param = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0]
tsteps = range(tspan[1], tspan[2], length=datasize)

prob_trueode = ODEProblem(F!, u, tspan, param) #, callback=cb)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat=tsteps))

prob_size = size(ode_data, 1)
dudt2 = Lux.Chain(
    Lux.Dense(prob_size, prob_size^2, tanh),
    Lux.Dense(prob_size^2, prob_size^2, tanh),
    Lux.Dense(prob_size^2, prob_size^2, tanh),
    Lux.Dense(prob_size^2, prob_size))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
losses = Float64[]
callback = function (p, l, pred; doplot::Bool=true, loss_step::Int=50)
    push!(losses, l)
    if length(losses) % loss_step == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
        if doplot
            plt = scatter(ode_data', label=["data S" "data E" "data I" "data R" "data D"])
            plot!(plt, pred', label=["prediction S" "prediction E" "prediction I" "prediction R" "prediction D"])
            display(plot(plt))
        end
    end
    return false
end

pinit = ComponentArray(p)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob,
    ADAM(0.05),
    callback=callback,
    maxiters=1_000,
    allow_f_increases=false
)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false
)

function plot_loss(losses::Vector{Float64}, label::Vector{String}, iter::Int)
    plt = plot(
        1:iter,
        losses[1:iter],
        yaxis=:log10,
        xaxis=:log10,
        label=label[1],
        xlabel="Iterations",
        ylabel="Loss",
        color=:blue,
    )
    plot!(
        iter+1:length(losses),
        losses[iter+1:end],
        yaxis=:log10,
        xaxis=:log10,
        label=label[2],
        xlabel="Iterations",
        ylabel="Loss",
        color=:red,
    )
    return plt
end
