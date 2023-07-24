# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

using Lux, Optimization, OptimizationOptimJL, DifferentialEquations
using SciMLSensitivity, ComponentArrays, OptimizationOptimisers
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse
using Distributions, Random, Plots, LinearAlgebra, Statistics, Zygote
using Dates

gr()

using DiffEqFlux


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
    du[3] = σ * E - γ * I - δ * I - μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R - μ * R # dR
    du[5] = δ * γ * I # dD
end

function get_data(;
    u::Vector{Float64}=[0.999, 0.0, 0.001, 0.0, 0.0],
    p::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.01],
    tspan::Tuple=(0.0, 30.0),
    rng::AbstractRNG=rng,
    f=F!
)
    prob = ODEProblem(f, u, tspan, p)
    solution = solve(
        prob,
        OrdinaryDiffEq.Vern7(),
        abstol=1e-12,
        reltol=1e-12,
        saveat=3
    )
    X = Array(solution)
    t = solution.t
    noisy_data = X + Float32(5e-3) * randn(rng, eltype(X), size(X))
    plot(solution, label=["True S" "True E" "True I" "True R" "True D"])
    display(scatter!(t, noisy_data', label=["Noisy S" "Noisy E" "Noisy I" "Noisy R" "Noisy D"]))
    return noisy_data, t
end

# rng = Xoshiro(42)
# u = [0.999, 0.0, 0.001, 0.0, 0.0]
# p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
# tspan = (0.0, 42.0)
# X, t = get_data(; u=u, p=p_true, tspan=tspan, rng=rng)

function nn_ode(
    data::Array,
    tspan::Tuple,
    activation_function=relu,
    maxiters=1000,
    doplot::Bool=false,
    saveat::Vector{Float64}=[0.0:1.0:42.0],
    seed::Int=42
)
    rng = Xoshiro(seed)
    # Multilayer FeedForward
    U = Lux.Chain(Lux.Dense(5, 64, activation_function), Lux.Dense(64, 5))
    # Get the initial parameters and state variables of the model
    p, st = Lux.setup(rng, U)
    prob_neuralode = NeuralODE(U, tspan, Tsit5(), saveat=saveat)

    function predict_neuralode(p)
        Array(prob_neuralode(u, p, st)[1])
    end

    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, data .- pred)
        return loss, pred
    end

    losses = Float64[]
    callback = function (p, l, pred; doplot=false)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        # plot current prediction against data
        if doplot
            plt = scatter(t, data', label=["S Measurements" "E Measurements" "I Measurements" "R Measurements" "D Measurements"])
            plot!(plt, t, pred', label=["S Prediction" "E Prediction" "I Prediction" "R Prediction" "D Prediction"])
            display(plot(plt))
        end
        return false
    end

    pinit = ComponentArray(p)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

    try
        optprob = Optimization.OptimizationProblem(optf, pinit)
        result_neuralode = Optimization.solve(
            optprob,
            ADAM(0.05),
            callback=callback,
            maxiters=trunc(Int, maxiters * 4 / 5)
        )
        optprob2 = remake(optprob, u0=result_neuralode.u)
        result_neuralode2 = Optimization.solve(
            optprob2,
            Optim.BFGS(initial_stepnorm=0.01),
            callback=callback,
            maxiters=trunc(Int, maxiters / 5)
        )
    catch ex
        @debug ex
    end

    callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=doplot)
    plt = doplot ? plot_loss(losses, ["ADAM", "BFGS"], maxiters) : nothing
    X̂ = predict_neuralode(result_neuralode2.u)
    Ŷ = U(X̂, result_neuralode2.u, st)[1]
    return X̂, Ŷ, plt
end

# LoadError: UndefVarError: `@variables` not defined
function sindy_like(
    X̂::Matrix,
    Ŷ::Matrix,
    tspan::StepRangeLen;
    p_true::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.01],
    opt=ADMM,
    λ=exp10.(-3:0.01:3),
    optimizers::Vector=[Optim.BFGS],
    seed::Int=42
)

    rng = Xoshiro(seed)

    @variables u[1:3]
    b = polynomial_basis(u, 5)
    basis = Basis(b, u)

    nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
    opt = opt(λ)

    options = DataDrivenCommonOptions(
        maxiters=10_000,
        normalize=DataNormalization(ZScoreTransform),
        selector=bic,
        digits=1,
        data_processing=DataProcessing(
            split=0.9,
            batchsize=30,
            shuffle=true,
            rng=rng,
        ),
    )

    nn_res = solve(nn_problem, basis, opt, options=options)
    nn_eqs = get_basis(nn_res)
    println(nn_res)

    # Define the recovered, hybrid model
    function recovered_dynamics!(du, u, p, t, p_true)
        û = nn_eqs(u, p) # Recovered equations
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ = p_true
        μ = δ / 1111
        du[1] = -û[1] * S # dS
        du[2] = û[1] * S - σ * E # dE
        du[3] = σ * E - γ * I - δ * I # dI
        du[4] = (1 - δ) * γ * I - ω * R # dR
        du[5] = δ * γ * I # dD
    end

    recovered_dynamics!(du, u, p, t) = recovered_dynamics!(du, u, p, t, p_true)

    estimation_prob = ODEProblem(
        recovered_dynamics!,
        u,
        (tspan[1], tspan[end]),
        get_parameter_values(nn_eqs),
    )
    estimate =
        solve(estimation_prob, Tsit5(; thread=OrdinaryDiffEq.True()), saveat=solution.t)

    # Plot
    # plot(solution)
    # plot!(estimate)

    function parameter_loss(p)
        Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
        sum(abs2, Ŷ .- Y)
    end

    optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
    parameter_res = Optimization.solve(optprob, Optim.BFGS(), maxiters=1000)

    # Look at long term prediction
    t_long = (0.0, tspan[2] * 2)
    estimation_prob = ODEProblem(recovered_dynamics!, u, t_long, parameter_res)
    estimate_long =
        solve(estimation_prob, Tsit5(; thread=OrdinaryDiffEq.True()), saveat=1) # Using higher tolerances here results in exit of julia
    # plot(estimate_long)

    # true_prob = ODEProblem(F!, u, t_long, p_true)
    # true_solution_long =
    #     solve(true_prob, Tsit5(; thread=OrdinaryDiffEq.True()), saveat=estimate_long.t)
    # plot!(true_solution_long)

    return estimate_long
end
