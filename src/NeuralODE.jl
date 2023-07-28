# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq,
    Plots, CUDA, SciMLSensitivity, Random, ComponentArrays, BSON, OptimizationOptimJL,
    ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse
import DiffEqFlux: NeuralODE

gr()

function set_backend!()
    try
        Lux.gpu_backend!(CUDA)
        @info "GPU backend set"
    catch ex
        @error ex
    end
end

function plot_loss(losses::Vector, label::Vector{String}, iter::Int)
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
    datasize::Int=30,
    rng::AbstractRNG=rng,
    f=F!,
    doplot::Bool=false
)
    prob = ODEProblem(f, u, tspan, p)
    solution = solve(prob, OrdinaryDiffEq.Tsit5(), saveat=range(tspan[1], tspan[2], length=datasize))
    X = Array(solution)
    t = solution.t
    noisy_data = X + Float32(5e-3) * randn(rng, eltype(X), size(X))
    if doplot
        plot(solution, label=["True S" "True E" "True I" "True R" "True D"])
        display(scatter!(t, noisy_data', label=["Noisy S" "Noisy E" "Noisy I" "Noisy R" "Noisy D"]))
    end
    return noisy_data, t
end

rng = Xoshiro(42)
u = [0.99, 0.0, 0.01, 0.0, 0.0]
p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
tspan = (0.0, 100.0)
datasize = Int(tspan[end])
X, t = get_data(; u=u, p=p_true, tspan=tspan, datasize=datasize, rng=rng, doplot=true)

X1 = X[:, 1:21]
x, y, plt = nn_ode(X1, tspan; maxiters=1000, doplot=true, saveat=0.0:1.0:20.0)
plt

# broken
res, plt = sindy_forecast(x, y, tspan, datasize; u0=X1[:, 1], doplot=true, saveat=0.0:1.0:20.0)

function forecast(
    data::Array,
    tspan::Tuple,
    datasize::Int,
    timeframe_to_forecast::Int;
    activation_function=relu,
    maxiters=1000,
    doplot::Bool=false,
    seed::Int=42
)
    set_backend!()
    saveat = range(tspan[1], tspan[2], length=datasize)
    X̂, Ŷ, plt = nn_ode(data, tspan; activation_function=activation_function, maxiters=maxiters, doplot=doplot, saveat=saveat, seed=seed)
    result, plt = sindy_forecast(X̂, Ŷ, tspan, timeframe_to_forecast; u0=data[:, 1], maxiters=maxiters, doplot=doplot)
    return result, plt
end

function datadriven_ode(data::Array, time)
    ddproblem = ContinuousDataDrivenProblem(data, time, GaussianKernel())
    @parameters t
    Symbolics.@variables u(t)[1:5]
    Ψ = Basis([u; u[1] * u[3]], u, independent_variable=t)
    res = solve(ddproblem, Ψ, STLSQ())
    return res
end

function nn_ode(
    data::Array,
    tspan::Tuple;
    activation_function=relu,
    maxiters::Int=1000,
    doplot::Bool=false,
    saveat::StepRangeLen,
    seed::Int=42
)
    CUDA.allowscalar(false) # Makes sure no slow operations are occuring
    rng = Xoshiro(seed)

    u = data[:, 1] |> Lux.gpu_device()
    data = data |> Lux.gpu_device()

    # Multilayer FeedForward
    U = Lux.Chain(Lux.Dense(5, 64, activation_function), Lux.Dense(64, 5))
    # Get the initial parameters and state variables of the model
    p, st = Lux.setup(rng, U)
    p = p |> ComponentArray |> Lux.gpu_device()
    st = st |> Lux.gpu_device()

    prob_neuralode = NeuralODE(U, tspan, Tsit5(), saveat=saveat)
    function predict_neuralode(p)
        first(prob_neuralode(u, p, st)) |> Lux.gpu_device()
    end

    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, data .- pred)
        return loss, pred
    end

    losses = Float32[]
    callback = function (p, l, pred; doplot=false)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        # plot current prediction against data
        if doplot
            plt = scatter(saveat, Array(data)', label=["S Measurements" "E Measurements" "I Measurements" "R Measurements" "D Measurements"])
            plot!(plt, saveat, Array(pred)', lw=3, label=["S NeuralODE" "E NeuralODE" "I NeuralODE" "R NeuralODE" "D NeuralODE"])
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

function save_model!(model, path::String)
    model = model |> Lux.cpu_device()
    BSON.@save path model
    return true
end

function sindy_forecast(
    X̂::Matrix,
    Ŷ::Matrix,
    tspan::Tuple,
    timeframe_to_forecast::Int;
    u0::Vector{Float64},
    saveat::StepRangeLen,
    maxiters::Int=1000,
    doplot::Bool=false
)
    @parameters t
    Symbolics.@variables u(t)[1:5]
    basis = Basis([u; u[1] * u[3]], u, independent_variable=t)

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
    # I do not write this code so i don't know why it broken up like this
    # ERROR: DimensionMismatch: arrays could not be broadcast to a common size
    nn_res = solve(nn_problem, basis, opt, options=options)
    nn_eqs = get_basis(nn_res)
    println(nn_res)

    # Define the recovered, hybrid model
    function recovered_dynamics!(du, u, p, t)
        û = nn_eqs(u, p) # Recovered equations
        du[1] = û[1]
        du[2] = û[2]
        du[3] = û[3]
        du[4] = û[4]
        du[5] = û[5]
    end

    estimation_prob = ODEProblem(recovered_dynamics!, u0, tspan, get_parameter_values(nn_eqs))
    estimate = solve(estimation_prob, Tsit5(), saveat=saveat)
    # Plot
    plot(solution)
    plot!(estimate)

    function parameter_loss(p)
        Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
        sum(abs2, Ŷ .- Y)
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
    parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters=maxiters)

    # Look at long term prediction
    t_long = (0.0, tspan[end] + timeframe_to_forecast)
    estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameter_res)
    estimate_long = solve(estimation_prob, Tsit5(), saveat=1) # Using higher tolerances here results in exit of julia
    plt = doplot ? plot(estimate_long) : nothing
    return estimate_long, plt
end
