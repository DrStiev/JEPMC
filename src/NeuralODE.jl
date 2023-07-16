# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/
# TODO: https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/
# TODO: https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL
using SciMLSensitivity, ComponentArrays, OptimizationOptimisers
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse
using Distributions, Random, Plots, LinearAlgebra, Statistics, Zygote

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
    R₀, γ, σ, ω, δ, η, ξ = p
    du[1] = -R₀ * γ * (1 - η) * S * I + ω * R - ξ * S # dS
    du[2] = R₀ * γ * (1 - η) * S * I - σ * E # dE
    du[3] = σ * E - γ * I - δ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R + ξ * S # dR
    du[5] = δ * γ * I # dD
end

# TODO: definire funzione comprensiva
function neural_ode(
    data::Matrix{Float64},
    p_true::Vector{Float64},
    tspan::Tuple;
    optimizers::Vector=[ADAM, Optim.BFGS],
    optimizersparameters::Vector{Float64}=[0.1, 0.01],
    maxiters::Vector{Int}=[5000, 1000],
    doplot::Bool=false,
    seed::Int=1234
)

    rng = Xoshiro(seed)

    prob_size = size(data, 1)
    U = Lux.Chain(
        Lux.Dense(prob_size, prob_size^2 * 2, tanh),
        Lux.Dense(prob_size^2 * 2, prob_size))
    p, st = Lux.setup(rng, U)

    function ude_dynamics!(du, u, p, t, p_true)
        û = U(u, p, st)[1]
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ, η, ξ = p_true
        du[1] = (-R₀ * γ * (1 - η) * S * I + ω * R - ξ * S) * û[1] # dS
        du[2] = (R₀ * γ * (1 - η) * S * I - σ * E) * û[2] # dE
        du[3] = (σ * E - γ * I - δ * I) * û[3] # dI
        du[4] = ((1 - δ) * γ * I - ω * R + ξ * S) * û[4] # dR
        du[5] = (δ * γ * I) * û[5] # dD
    end

    nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_true)
    prob_nn = ODEProblem(nn_dynamics!, data[:, 1], tspan, p)

    function predict(θ, X=data[:, 1], T=tspan)
        _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
        Array(solve(_prob, Tsit5(), saveat=T, abstol=1e-6, reltol=1e-6, verbose=false))
    end

    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, X .- X̂)
    end

    losses = Float64[]
    callback = function (p, l; loss_step::Int=100)
        push!(losses, l)
        if length(losses) % loss_step == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    res1 = Optimization.solve(optprob, optimizers[1](optimizersparameters[1]), callback=callback, maxiters=maxiters[1])
    println("Training loss after $(length(losses)) iterations: $(losses[end])")

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, optimizers[2](optimizersparameters[2]), callback=callback, maxiters=maxiters[2])
    println("Final training loss after $(length(losses)) iterations: $(losses[end])")

    plt = nothing
    if doplot
        # TODO: make plot
        X̂ = predict(res2.u, data[:, 1], tspan)
        # Trained on noisy data vs real solution
        pl_trajectory = plot(tspan, transpose(X̂), xlabel="t", ylabel="percentage", label=["S Approximation" "E Approximation" "I Approximation" "R Approximation" "D Approximation"])
        scatter!(tspan, transpose(data), label=["S Measurements" "E Measurements" "I Measurements" "R Measurements" "D Measurements"])
        # Ideal unknown interactions of the predictors
        Ȳ = []
        Ŷ = U(X̂, res2.u, st)[1]
        # Plot the error
        pl_reconstruction_error = plot(tspan, norm.(eachcol(Ȳ .- Ŷ)), yaxis=:log, xlabel="t", ylabel="L2-Error", label=nothing, color=:red)
        pl_missing = plot(pl_recontruction, pl_recontruction_error, layout=(1, 2))
        plt = plot(pl_trajectory, pl_missing)
    end
    return (res2.u, Ŷ=U(X̂, res2.u, st)[1], losses, plt)
end

function sparse_regression(
    X̂::Matrix{Float64},
    Ŷ::Matrix{Float64},
    p_true::Vector{Float64},
    tspan::Tuple;
    λ=exp10.(-3:0.01:3),
    optimizers::Vector=[ADMM, Optim.BFGS],
    maxiters::Int=1000,
    doplot::Bool=false,
    seed::Int=1234
)
    @variables u[1:size(X̂, 1)]
    b = polynomial_basis(u, size(X̂, 1))
    basis = Basis(b, u)

    nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
    opt = optimizers[1](λ)

    options = DataDrivenCommonOptions(maxiters=10_000,
        normalize=DataNormalization(ZScoreTransform),
        selector=bic, digits=1,
        data_processing=DataProcessing(split=0.9,
            batchsize=30,
            shuffle=true,
            rng=Xoshiro(seed)))

    nn_res = solve(nn_problem, basis, opt, options=options)
    nn_eqs = get_basis(nn_res)
    println(nn_res)

    # Define the recovered, hybrid model
    function recovered_dynamics!(du, u, p, t, p_true)
        û = nn_eqs(u, p) # recovered equations
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ, η, ξ = p_true
        du[1] = (-R₀ * γ * (1 - η) * S * I + ω * R - ξ * S) * û[1] # dS
        du[2] = (R₀ * γ * (1 - η) * S * I - σ * E) * û[2] # dE
        du[3] = (σ * E - γ * I - δ * I) * û[3] # dI
        du[4] = ((1 - δ) * γ * I - ω * R + ξ * S) * û[4] # dR
        du[5] = (δ * γ * I) * û[5] # dD
    end

    recovered_dynamics!(du, u, p, t) = recovered_dynamics!(du, u, p, t, p_true)
    estimation_prob = ODEProblem(recovered_dynamics!, X̂[:, 1], tspan, get_parameter_values(nn_eqs))
    estimate = solve(estimation_prob, Tsit5(), saveat=tspan)

    function parameter_loss(p)
        Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
        sum(abs2, Ŷ .- Y)
    end

    optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
    parameter_res = Optimization.solve(optprob, optimizers[2], maxiters=maxiters)

    # Simulation long term
    t_long = (0.0, 50.0)
    estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameter_res)
    estimate_long = solve(estimation_prob, Tsit5(), saveat=tsteps)

    return estimate_long
end

rng = Xoshiro(1234)
u = [(1e6 - 1) / 1e6, 0, 1 / 1e6, 0, 0]
datasize = 171 # ≈ 1 entrata a settimana
tspan = (0.0f0, 1200.0f0)
param = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0]
tsteps = range(tspan[1], tspan[2], length=datasize)

prob_trueode = ODEProblem(F!, u, tspan, param)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat=tsteps, abstol=1e-12, reltol=1e-12))

x, y, z, plt = neural_ode(ode_data, param, tspan)
