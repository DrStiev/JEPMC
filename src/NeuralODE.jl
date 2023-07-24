# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

using Lux, Optimization, OptimizationOptimJL, DifferentialEquations
using SciMLSensitivity, ComponentArrays, OptimizationOptimisers
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse
using Distributions, Random, Plots, LinearAlgebra, Statistics, Zygote
using Dates, DiffEqFlux, CUDA

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
    solution = solve(prob, OrdinaryDiffEq.Tsit5(), saveat=1)
    X = Array(solution)
    t = solution.t
    noisy_data = X + Float32(5e-3) * randn(rng, eltype(X), size(X))
    plot(solution, label=["True S" "True E" "True I" "True R" "True D"])
    display(scatter!(t, noisy_data', label=["Noisy S" "Noisy E" "Noisy I" "Noisy R" "Noisy D"]))
    return noisy_data, t
end

# TODO: capire come fare predizioni

rng = Xoshiro(42)
u = [0.99, 0.0, 0.01, 0.0, 0.0]
p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
tspan = (0.0, 100.0)
X, t = get_data(; u=u, p=p_true, tspan=tspan, rng=rng)

X
t
t[1:22]
X[:, 21:42]
x, y, plt = nn_ode(X[:, 21:42], (0.0, 51.0); saveat=t[1:22], maxiters=2000)

x
y
plt

x, y, plt = nn_ode_gpu(X[:, 21:42], (0.0, 51.0); saveat=t[1:22], maxiters=2000, doplot=true)
x
y
plt

function nn_ode_gpu(
    data::Array,
    tspan::Tuple;
    activation_function=relu,
    maxiters=1000,
    doplot::Bool=false,
    saveat::Vector{Float64}=[0.0:1.0:size(data, 2)],
    seed::Int=42
)
    CUDA.allowscalar(false) # Makes sure no slow operations are occuring
    rng = Xoshiro(seed)
    u = data[:, 1] |> Lux.gpu
    data = data |> Lux.gpu
    # Multilayer FeedForward
    U = Lux.Chain(Lux.Dense(5, 64, activation_function), Lux.Dense(64, 5))
    # Get the initial parameters and state variables of the model
    p, st = Lux.setup(rng, U)
    p = p |> ComponentArray |> Lux.gpu
    st = st |> Lux.gpu
    prob_neuralode = NeuralODE(U, tspan, Tsit5(), saveat=saveat)

    function predict_neuralode(p)
        Lux.gpu(first(prob_neuralode(u, p, st)))
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
            plt = scatter(saveat, Array(data)', label=["S Measurements" "E Measurements" "I Measurements" "R Measurements" "D Measurements"])
            plot!(plt, saveat, Array(pred)', lw=3, label=["S Prediction" "E Prediction" "I Prediction" "R Prediction" "D Prediction"])
            display(plot(plt))
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    result_neuralode = Optimization.solve(
        optprob,
        ADAM(0.05),
        callback=callback,
        maxiters=trunc(Int, maxiters * 4 / 5)
    )
    callback(result_neuralode.u, loss_neuralode(result_neuralode.u)...; doplot=true)

    optprob2 = remake(optprob, u0=result_neuralode.u)
    result_neuralode2 = Optimization.solve(
        optprob2,
        Optim.BFGS(initial_stepnorm=0.01),
        callback=callback,
        maxiters=trunc(Int, maxiters / 5)
    )
    callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)

    plt = doplot ? plot_loss(losses, ["ADAM", "BFGS"], trunc(Int, maxiters * 4 / 5)) : nothing
    X̂ = predict_neuralode(result_neuralode2.u)
    Ŷ = U(X̂, result_neuralode2.u, st)[1]
    return X̂, Ŷ, plt
end

function nn_ode(
    data::Array,
    tspan::Tuple;
    activation_function=relu,
    maxiters=1000,
    doplot::Bool=false,
    saveat::Vector{Float64}=[0.0:1.0:size(data, 2)],
    seed::Int=42
)
    rng = Xoshiro(seed)
    u = data[:, 1]
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
            plt = scatter(saveat, data', label=["S Measurements" "E Measurements" "I Measurements" "R Measurements" "D Measurements"])
            plot!(plt, saveat, pred', lw=3, label=["S Prediction" "E Prediction" "I Prediction" "R Prediction" "D Prediction"])
            display(plot(plt))
        end
        return false
    end

    pinit = ComponentArray(p)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    result_neuralode = Optimization.solve(
        optprob,
        ADAM(0.05),
        callback=callback,
        maxiters=trunc(Int, maxiters * 4 / 5)
    )
    callback(result_neuralode.u, loss_neuralode(result_neuralode.u)...; doplot=true)

    optprob2 = remake(optprob, u0=result_neuralode.u)
    result_neuralode2 = Optimization.solve(
        optprob2,
        Optim.BFGS(initial_stepnorm=0.01),
        callback=callback,
        maxiters=trunc(Int, maxiters / 5)
    )
    callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)

    plt = doplot ? plot_loss(losses, ["ADAM", "BFGS"], maxiters) : nothing
    X̂ = predict_neuralode(result_neuralode2.u)
    Ŷ = U(X̂, result_neuralode2.u, st)[1]
    return X̂, Ŷ, plt
end
