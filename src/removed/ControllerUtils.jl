using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, CUDA
using LinearAlgebra, Statistics, Plots
using ComponentArrays, Lux, Zygote, StableRNGs, DataFrames

include("Utils.jl")

function ude_prediction(
    data::DataFrame,
    p_true::Array{Float64},
    tshift::Int;
    seed::Int=1234,
    plotLoss::Bool=false,
    maxiters::Int=5000,
    optimizers::Vector=[ADAMW, Optim.LBFGS], # cercare gli optimizer
    activation_function=tanh,
    lossTitle::String="loss",
    verbose::Bool=false
)
    # https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/
    # https://docs.sciml.ai/Optimization/stable/optimization_packages/optimisers/
    X = Array(data)'
    tspan = float.([i for i = 1:size(Array(data), 1)])
    timeshift = float.([i for i = 1:tshift])
    if any(X .>= 1)
        X = X ./ sum(data[1, :])
    end

    s = size(X, 1)

    U = Lux.Chain(
        Lux.Dense(s, s^2, activation_function),
        Lux.Dense(s^2, s^2, activation_function),
        Lux.Dense(s^2, s^2, activation_function),
        Lux.Dense(s^2, s),
    )
    p, st = Lux.setup(StableRNG(seed), U)

    function ude_dynamics!(du, u, p, t, p_true)
        û = U(u, p, st)[1]
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ, η, ξ = p_true
        du[1] = ((-R₀ * γ * (1 - η) * S * I / (S + E + I + R)) + (ω * R)) * û[1] # dS
        du[2] = ((R₀ * γ * (1 - η) * S * I / (S + E + I + R)) - (σ * E)) * û[2] # dE
        du[3] = ((σ * E) - (γ * I) - (δ * I)) * û[3] # dI
        du[4] = (((1 - δ) * γ * I) - (ω * R)) * û[4] # dR
        du[5] = (δ * I * γ) * û[5] # dD
    end

    condition_voc(u, t, integrator) = rand(rng) < 8e-3
    function affect_voc!(integrator)
        println("voc")
        integrator.p_true[1] = rand(rng, Uniform(3.3, 5.7))
        integrator.p_true[2] = abs(rand(rng, Normal(integrator.p_true[2], integrator.p_true[2] / 10)))
        integrator.p_true[3] = abs(rand(rng, Normal(integrator.p_true[3], integrator.p_true[3] / 10)))
        integrator.p_true[4] = abs(rand(rng, Normal(integrator.p_true[4], integrator.p_true[4] / 10)))
        integrator.p_true[5] = abs(rand(rng, Normal(integrator.p_true[5], integrator.p_true[5] / 10)))
    end

    voc_cb = ContinuousCallback(condition_voc, affect_voc!)

    nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_true)
    prob_nn = ODEProblem(nn_dynamics!, X[:, 1], (tspan[1], tspan[end]), p, callback=voc_cb)

    function predict(θ, X=X[:, 1], T=tspan)
        _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
        Array(
            solve(
                _prob,
                Vern7(; thread=OrdinaryDiffEq.True()),
                saveat=T,
                abstol=1e-6,
                reltol=1e-6,
                verbose=verbose,
            ),
        )
    end

    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, X .- X̂)
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

    iterations = trunc(Int, maxiters / 5)
    res1 = Optimization.solve(
        optprob,
        optimizers[1](),
        callback=callback,
        maxiters=iterations * 4,
    )
    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(
        optprob2,
        optimizers[2](),
        callback=callback,
        maxiters=iterations,
    )

    if plotLoss
        pl_losses = plot(
            1:iterations,
            losses[1:iterations],
            yaxis=:log10,
            xaxis=:log10,
            xlabel="Iterations",
            ylabel="Loss",
            label="ADAM",
            color=:blue,
        )
        plot!(
            iterations+1:length(losses),
            losses[iterations+1:end],
            yaxis=:log10,
            xaxis=:log10,
            xlabel="Iterations",
            ylabel="Loss",
            label="LBFGS",
            color=:red,
        )

        save_plot(pl_losses, "img/prediction/", lossTitle * "_NN_", "pdf")
    end

    p_trained = res2.u

    ts = first(timeshift):(mean(diff(timeshift))):last(timeshift)
    X̂ = predict(p_trained, X[:, 1], ts)
    Ŷ = U(X̂, p_trained, st)[1]
    long_pred = nothing
    try
        long_pred = symbolic_regression(
            X̂,
            Ŷ,
            p_true,
            tshift;
            optimizers=optimizers,
            verbose=verbose,
            plotLoss=plotLoss,
            lossTitle=lossTitle,
            maxiters=maxiters * 2
        )
    catch ex
        isdir("data/error/") == false && mkpath("data/error/")
        joinpath("data/error/", "log_" * string(today()) * ".txt")
        log = @error "Symbolic regression failed" exception = (ex, catch_backtrace())
        open("data/error/log_" * string(today()) * ".txt", "a") do io
            write(io, log)
        end
    finally
        return (X̂, Ŷ, long_pred)
    end
end

function symbolic_regression(
    X̂::Matrix,
    Ŷ::Matrix,
    p_true::Vector{Float64},
    timeshift::Int;
    opt=ADMM,
    λ=exp10.(-3:0.01:3),
    optimizers::Vector=[ADAMW, Optim.LBFGS],
    maxiters::Int=10_000,
    seed::Int=1234,
    verbose::Bool=false,
    plotLoss::Bool=false,
    lossTitle::String="title"
)

    tspan = (1.0, float(timeshift))
    u0 = X̂[:, 1]
    nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
    opt = opt(λ)
    @variables u[1:size(X̂, 1)]
    b = polynomial_basis(u, size(X̂, 1))
    basis = Basis(b, u)

    options = DataDrivenCommonOptions(
        maxiters=maxiters,
        normalize=DataNormalization(ZScoreTransform),
        selector=bic,
        digits=1,
        data_processing=DataProcessing(
            split=0.9,
            batchsize=30,
            shuffle=true,
            rng=StableRNG(seed),
        ),
    )

    nn_res = solve(nn_problem, basis, opt, options=options, verbose=verbose)
    nn_eqs = get_basis(nn_res)

    function recovered_dynamics!(du, u, p, t, p_true)
        û = nn_eqs(u, p)
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ, η, ξ = p_true
        du[1] = ((-R₀ * γ * (1 - η) * S * I / (S + E + I + R)) + (ω * R)) * û[1] # dS
        du[2] = ((R₀ * γ * (1 - η) * S * I / (S + E + I + R)) - (σ * E)) * û[2] # dE
        du[3] = ((σ * E) - (γ * I) - (δ * I)) * û[3] # dI
        du[4] = (((1 - δ) * γ * I) - (ω * R)) * û[4] # dR
        du[5] = (δ * I * γ) * û[5] # dD
    end

    condition_voc(u, t, integrator) = rand(rng) < 8e-3
    function affect_voc!(integrator)
        println("voc")
        integrator.p_true[1] = rand(rng, Uniform(3.3, 5.7))
        integrator.p_true[2] = abs(rand(rng, Normal(integrator.p_true[2], integrator.p_true[2] / 10)))
        integrator.p_true[3] = abs(rand(rng, Normal(integrator.p_true[3], integrator.p_true[3] / 10)))
        integrator.p_true[4] = abs(rand(rng, Normal(integrator.p_true[4], integrator.p_true[4] / 10)))
        integrator.p_true[5] = abs(rand(rng, Normal(integrator.p_true[5], integrator.p_true[5] / 10)))
    end

    voc_cb = ContinuousCallback(condition_voc, affect_voc!)

    dynamics!(du, u, p, t) = recovered_dynamics!(du, u, p, t, p_true)
    estimation_prob = ODEProblem(dynamics!, u0, tspan, get_parameter_values(nn_eqs), callback=voc_cb)
    estimate = solve(
        estimation_prob,
        Tsit5(; thread=OrdinaryDiffEq.True()),
        saveat=1.0,
        verbose=verbose,
    )

    function parameter_loss(p)
        Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
        sum(abs2, Ŷ .- Y)
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
    optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))

    iterations = trunc(Int, maxiters / 5)
    res1 = Optimization.solve(
        optprob,
        optimizers[1](),
        callback=callback,
        maxiters=iterations * 4,
    )

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    parameter_res = Optimization.solve(
        optprob2,
        optimizers[2](),
        callback=callback,
        maxiters=iterations,
    )

    if plotLoss
        pl_losses = plot(
            1:iterations,
            losses[1:iterations],
            yaxis=:log10,
            xaxis=:log10,
            xlabel="Iterations",
            ylabel="Loss",
            label="ADAM",
            color=:blue,
        )
        plot!(
            iterations+1:length(losses),
            losses[iterations+1:end],
            yaxis=:log10,
            xaxis=:log10,
            xlabel="Iterations",
            ylabel="Loss",
            label="LBFGS",
            color=:red,
        )

        save_plot(pl_losses, "img/prediction/", lossTitle * "_SR_", "pdf")
    end

    estimation_prob = ODEProblem(dynamics!, u0, tspan, parameter_res, callback=voc_cb)
    estimate_long = solve(
        estimation_prob,
        Tsit5(; thread=OrdinaryDiffEq.True()),
        saveat=1.0,
        verbose=verbose,
    ) # Using higher tolerances here results in exit of julia
    return estimate_long
end
