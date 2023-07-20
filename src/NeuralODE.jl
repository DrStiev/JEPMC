# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL
using SciMLSensitivity, ComponentArrays, OptimizationOptimisers, Flux
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse
using Distributions, Random, Plots, LinearAlgebra, Statistics, Zygote
using Dates

gr()

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

function F!(du, u, p, t)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ = p
    μ = δ / 1111
    du[1] = μ * sum(u) - R₀ * γ * S * I + ω * R - μ * S # dS
    du[2] = R₀ * γ * S * I - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I * μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R - μ * R # dR
    du[5] = δ * γ * I # dD
end

rng = Xoshiro(1234)
u = [0.99, 0.0, 0.01, 0.0, 0.0]
p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.0007]
tspan = (0.0, 100.0)

prob = ODEProblem(F!, u, tspan, p_true)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=1)
X = Array(solution)
t = solution.t

noisy_data = X + Float32(1e-2) * randn(rng, eltype(X), size(X))
plot(abs.(X - noisy_data)')

plot(solution, label=["True S" "True E" "True I" "True R" "True D"])
scatter!(noisy_data', label=["Noisy S" "Noisy E" "Noisy I" "Noisy R" "Noisy D"])

# Multilayer FeedForward
U = Lux.Chain(
    Lux.Dense(3, 64, relu),
    Lux.Dense(64, 64, relu),
    Lux.Dense(64, 1)
)
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)

function sir_ude!(du, u, p, t, p_true)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ = p_true
    λ = U([S, I, D], p, st)[1]
    μ = δ / 101
    du[1] = μ * sum(u) - λ[1] * S - μ * S # dS
    du[2] = λ[1] * S - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I * μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R - μ * R # dR
    du[5] = δ * γ * I # dD
end

nn_dynamics!(du, u, p, t) = sir_ude!(du, u, p, t, p_true)
prob_nn = ODEProblem(nn_dynamics!, noisy_data[:, 1], tspan, p)

function predict(θ, X=noisy_data[:, 1], T=t)
    _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
    Array(solve(_prob, Vern7(), saveat=T,
        abstol=1e-6, reltol=1e-6, verbose=false))
end

# poisson loss as we are comparing our model against counts of new cases
function loss(θ)
    pred = predict(θ)
    println("$(size(pred))")
    mean(abs2, noisy_data .- pred)
end

losses = Float64[]
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 100 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

maxiters = 5000
res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=maxiters)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = trun(Int,maxiters/5))
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

plt = plot_loss(losses, ["ADAM, LBFGS"], maxiters)
include("Utils.jl")
save_plot(plt, "img/loss/" * string(today()) * "/", "UDE_LOSS", "pdf")

@variables u[1:3]
b = polynomial_basis(u)
basis = Basis(b, u);

ts = first(t):(mean(diff(t))/2):last(t)
X̂ = predict(res2.u, noisy_data[:,1], ts)
Ŷ = U(X̂, res2.u, st)[1]

nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
λ = exp10.(-3:0.01:3)
opt = ADMM(λ)

options = DataDrivenCommonOptions(maxiters=10_000,
    normalize=DataNormalization(ZScoreTransform),
    selector=bic, digits=1,
    data_processing=DataProcessing(split=0.9,
        batchsize=30,
        shuffle=true,
        rng=rng))

nn_res = solve(nn_problem, basis, opt, options=options)
nn_eqs = get_basis(nn_res)
println(nn_res)

# Define the recovered, hybrid model
function recovered_dynamics!(du, u, p, t, p_true)
    û = nn_eqs(u, p) # Recovered equations
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ = p_true
    μ = δ / 101
    du[1] = μ * sum(u) - û[1] * S - μ * S # dS
    du[2] = û[1] * S - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I * μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R - μ * R # dR
    du[5] = δ * γ * I # dD
end

recovered_dynamics!(du, u, p, t) = recovered_dynamics!(du, u, p, t, p_true)

estimation_prob = ODEProblem(recovered_dynamics!, u, tspan, get_parameter_values(nn_eqs))
estimate = solve(estimation_prob, Tsit5(), saveat=solution.t)

# Plot
plot(solution)
plot!(estimate)

function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
    sum(abs2, Ŷ .- Y)
end

optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters=1000)

# Look at long term prediction
t_long = (0.0, tspan[2]*2)
estimation_prob = ODEProblem(recovered_dynamics!, u, t_long, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat=1) # Using higher tolerances here results in exit of julia
plot(estimate_long)

true_prob = ODEProblem(F!, u, t_long, p_true)
true_solution_long = solve(true_prob, Tsit5(), saveat=estimate_long.t)
plot!(true_solution_long)
