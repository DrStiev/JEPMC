# TODO: https://julialang.org/blog/2019/01/fluxdiffeq/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# TODO: https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# TODO: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq
using Plots, CUDA, SciMLSensitivity, Random, ComponentArrays, BSON, OptimizationOptimJL
using ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse, LuxCUDA
import DiffEqFlux: NeuralODE
import Statistics: mean

# https://docs.sciml.ai/SciMLSensitivity/stable/examples/sde/SDE_control/

# https://docs.juliahub.com/DiffEqFlux/BdO4p/1.10.1/examples/NeuralOptimalControl/
# https://arxiv.org/abs/2210.11245
# https://www.youtube.com/watch?v=omS3ZngEygw
# https://arxiv.org/pdf/2210.11245.pdf

# https://medium.com/swlh/neural-ode-for-reinforcement-learning-and-nonlinear-optimal-control-cartpole-problem-revisited-5408018b8d71
# https://github.com/paulxshen/neural-ode-cartpole/blob/master/Cartpole.jl

# https://sebastiancallh.github.io/post/neural-ode-weather-forecast/
# https://github.com/SebastianCallh/neural-ode-weather-forecast

# https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/training_tips/multiple_nn/
# https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/parameter_estimation_ode/

# https://arxiv.org/pdf/2012.06684.pdf
# https://github.com/samuela/ctpg

gr()
function set_backend!()
    try
        Lux.gpu_backend!(CUDA)
        @info "GPU backend set"
    catch ex
        @error ex
        @info "GPU backend not set"
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
    R₀, γ, σ, ω, δ, η, ξ = p
    μ = δ / 1111
    du[1] = μ * sum(u) - R₀ * γ * (1 - η) * S * I + ω * R - ξ * S - μ * S # dS
    du[2] = R₀ * γ * (1 - η) * S * I - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I - μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R + ξ * S - μ * R # dR
    du[5] = δ * γ * I # dD
end

function get_data(;
    u::Vector{Float64}=[0.999, 0.0, 0.001, 0.0, 0.0],
    p::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.01, 0.0, 0.0],
    tspan::Tuple=(0.0, 30.0),
    datasize::Int=30,
    rng::AbstractRNG=Random.default_rng(),
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

function forecast(
    data::Array,
    tspan::Tuple,
    datasize::Int,
    timeframe_to_forecast::Int;
    activation_function=relu,
    maxiters=1000,
    doplot::Bool=false,
    rng::AbstractRNG=Random.default_rng()
)
    set_backend!()
    saveat = range(tspan[1], tspan[2], length=datasize)
    X̂, Ŷ, plt = nn_ode(data, tspan; activation_function=activation_function, maxiters=maxiters, doplot=doplot, saveat=saveat, rng=rng)
    result, plt = sindy_forecast(X̂, Ŷ, tspan, timeframe_to_forecast; u0=data[:, 1], maxiters=maxiters, doplot=doplot, rng=rng)
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

function neural_ode(data_dim::Int, t; device=Lux.cpu_device(), rng::AbstractRNG=Random.default_rng())
    # Multilayer FeedForward
    U = Lux.Chain(
        Lux.Dense(data_dim, 32, swish),
        Lux.Dense(32, 32, swish),
        Lux.Dense(32, data_dim, tanh)
    )
    # Get the initial parameters and state variables of the model
    p, state = Lux.setup(rng, U)
    p = p |> ComponentArray |> device
    state = state |> device

    prob_neuralode = NeuralODE(U, extrema(Float32.(t)), Tsit5(), saveat=t, verbose=false)
    return prob_neuralode, p, state
end

function nn_ode(
    data::Array,
    t;
    maxiters::Int=1000,
    rng::AbstractRNG=Random.default_rng()
)
    CUDA.allowscalar(false) # Makes sure no slow operations are occuring
    device = CUDA.functional() ? Lux.gpu_device() : Lux.cpu_device()
    device = Lux.cpu_device()

    data = Float32.(data)
    u = data[:, 1] |> device
    data = data |> device

    prob_neuralode, p, st = neural_ode(size(data, 1), t; device=device, rng=rng)

    function predict_neuralode(p)
        first(prob_neuralode(u, p, st)) |> device
    end

    function loss_neuralode(p)
        pred = predict_neuralode(p)
        # handling Zygote's behaviour for zero gradients
        if size(pred) == size(data)
            loss = mean(abs2, data .- pred)
            return loss, pred
        else
            return Inf, pred
        end
    end

    losses = Float32[]
    callback = function (p, l; iter=50)
        push!(losses, l)
        if length(losses) % iter == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    res1 = Optimization.solve(
        optprob,
        ADAM(),
        callback=callback,
        maxiters=trunc(Int, maxiters * 4 / 5)
    )

    optprob2 = remake(optprob, u0=res1.u)
    res2 = Optimization.solve(
        optprob2,
        Optim.BFGS(initial_stepnorm=0.01),
        callback=callback,
        maxiters=trunc(Int, maxiters / 5)
    )
    callback(res2.u, loss_neuralode(res2.u)...; iter=1)
    return res2.minimizer, st, losses
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
    saveat::StepRangeLen,
    maxiters::Int=1000,
    rng::AbstractRNG=Random.default_rng()
)
    u0 = X̂[:, 1]
    saveat = isnothing(saveat) ? range(0.0, size(X̂, 2) - 1) : saveat
    @parameters t
    Symbolics.@variables u(t)[1:5]
    Ψ = Basis([u; u[1] * u[3]], u, independent_variable=t)

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
    nn_res = solve(nn_problem, Ψ, opt, options=options, verbose=false)
    nn_eqs = get_basis(nn_res)
    println(nn_res)
    prinltn(nn_eqs)
    println(get_parameter_map(nn_eqs))

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
    return estimate_long
end

rng = Xoshiro(1234)
u = [0.99, 0.0, 0.01, 0.0, 0.0]
p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01, 0.0, 0.0]
tspan = (0.0, 100.0)
datasize = Int(tspan[end])
X, t = get_data(; u=u, p=p_true, tspan=tspan, datasize=datasize, rng=rng, doplot=true)
X1 = X
res, state, losses = nn_ode(X1, 0.0:1.0:size(X1, 2)-1; maxiters=Int(size(X1, 2) * 20))
res
state
losses

prob_neuralode, p, st = neural_ode(size(X1, 1), 0.0:1.0:size(X1, 2)*8; device=Lux.cpu_device(), rng=rng)
y = first(prob_neuralode(X1[:, 1], res, st))
plot(y)

# broken
res, plt = sindy_forecast(res, state, (0.0, size(X1, 2)), Int(size(X1, 2) * 1.5))
