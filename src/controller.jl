module controller
# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# External Libraries
using ComponentArrays, Lux, Zygote

using DataFrames, Plots, Random, Agents, Distributions
using LinearAlgebra, Statistics, StableRNGs

# parametri su cui il controllore può agire:
# η → countermeasures (0.0 - 1.0)
# Rᵢ → objective value for R₀
# ξ → vaccination rate

# https://github.com/epirecipes/sir-julia
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_optimization/ode_lockdown_optimization.md
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ude/ude.md
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ddeq/ode_ddeq.md
function countermeasures!(model::StandardABM, data::DataFrame; β = 3, saveat = 3)
    # applico delle contromisure rozze per iniziare 
    # https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_optim/ode_optim.md
    # beta feature
    # NAR = model.node_at_risk(model)
    # model.reduce_migration_rate!(NAR)
    slope(x, β) = 1 / (1 + (x / (1 - x))^(-β)) # simil sigmoide.
    length(data[!, :infected_status]) == 0 && return
    # cappo il ratio tra [-1,1]
    ratio = length(data[!, 1]) / (data[end, :infected_status] - data[1, :infected_status])
    s = slope(abs(ratio), 3) / 3
    # rapida crescita, lenta decrescita
    if ratio > 0.0
        model.η = s ≥ model.η ? s : model.η * (1.0 + s)
    elseif ratio < 0.0
        model.η /= (1.0 + s)
    end
    model.η = model.η ≥ 1.0 ? 1.0 : model.η
    model.η = isnan(model.η) ? 0.0 : model.η
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
function predict(data::DataFrame, tspan::Tuple; seed = 1337)
    tspan = float.(tspan)
    rng = StableRNG(seed)
    Xₙ = float.(Array(data)')
    N = sum(Xₙ[:, 1])
    Xₙ = Xₙ ./ N
    rbf(x) = exp.(-(x .^ 2))

    # define our UDE. We will use Lux.jl to define the neural network
    # Multilayer FeedForward (5 states S,E,I,R,D and 8 hidden, activation tanh or rbf)
    U = Lux.Chain(
        Lux.Dense(5, 8, rbf),
        Lux.Dense(8, 8, rbf),
        Lux.Dense(8, 8, rbf),
        Lux.Dense(8, 5),
    )
    p, st = Lux.setup(rng, U)

    # we define the UDE as a dynamical system
    function ude_dynamics!(du, u, p, t)
        û = U(u, p, st)[1] # network prediction
        du[1] = û[1]
        du[2] = û[2]
        du[3] = û[3]
        du[4] = û[4]
        du[5] = û[5]
    end

    # Define the hybrid model
    # function ude_dynamics!(du, u, p, t, p_true)
    #     û = U(u, p, st)[1] # Network prediction
    #     du[1] = -p_true[1] * u[1] * p_true[2] * u[3] + û[1]
    #     du[2] = p_true[1] * u[1] * p_true[2] * u[3] - û[2]
    #     du[3] = û[2] - p_true[2] * u[3] + û[3]
    #     du[4] = p_true[2] * u[3] - û[1]
    #     du[5] = p_true[2] * u[3] * p_true[5]
    # end

    prob_nn = ODEProblem(ude_dynamics!, Xₙ[:, 1], tspan, p)

    # let's build a training loop around our UDE
    # function predict which runs our simulation at new
    # neural network weights. Reacall that weights are the parameters
    # of the ODE, so we want to update the parameters and then run 
    # again
    function predict(θ, X = Xₙ[:, 1], T = tspan)
        _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
        Array(solve(_prob, Vern7(), saveat = T, abstol = 1e-6, reltol = 1e-6))
    end

    # for our loss function we solve the ODE at our new parameters
    # and check its L2 loss against the dataset
    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, Xₙ .- X̂)
    end

    # last but not least we want to track out optimization to 
    # define a callback. 
    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    # now we are ready to train
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    # now we optimize the result using a mixed strategy. we start 
    # using some iterations of ADAM because it's better at finding
    # a good general area of parameter space. then we move to BFGS
    # which will quicly hone to a local minimum. Only use one of them
    # will ends up be bad for the UDEs.
    res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 5000)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")
    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 1000)
    println("Final training loss after $(length(losses)) iterations: $(losses[end])")

    # Rename the best candidate
    p_trained = res2.u
    ts = first(tspan):(mean(diff(tspan))/2):last(tspan)
    X̂ = predict(p_trained, Xₙ[:, 1], ts)
    Ŷ = U(X̂, p_trained, st)[1]

    @variables u[1:size(Xₙ)[1]]
    b = polynomial_basis(u, 4)
    basis = Basis(b, u)

    nn_problem = DirectDataDrivenProblem(X̂, Ŷ)

    λ = exp10.(-3:0.01:3)
    opt = ADMM(λ)

    options = DataDrivenCommonOptions(
        maxiters = 10_000,
        normalize = DataNormalization(ZScoreTransform),
        selector = bic,
        digits = 1,
        data_processing = DataProcessing(
            split = 0.9,
            batchsize = 30,
            shuffle = true,
            rng = StableRNG(1111),
        ),
    )

    nn_res = solve(nn_problem, basis, opt, options = options)
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

    estimation_prob =
        ODEProblem(recovered_dynamics!, u0, tspan, get_parameter_values(nn_eqs))
    estimate = solve(estimation_prob, Tsit5(), saveat = solution.t)

    function parameter_loss(p)
        Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
        sum(abs2, Ŷ .- Y)
    end

    optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
    parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 1000)

    t_long = (0.0, 50.0)
    estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameter_res)
    estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
end

# https://docs.sciml.ai/Overview/stable/showcase/optimization_under_uncertainty/
function policy!(model::StandardABM, data::DataFrame)
    # cerco di massimizzare la happiness e minimizzare gli infetti
end

function policy!(data::DataFrame; seed = 1337)
    # https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
    # https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
    rng = Xoshiro(seed)
end

end
