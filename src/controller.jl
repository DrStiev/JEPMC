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
function countermeasures!(model::StandardABM, data::DataFrame; β=3, saveat=3)
    # applico delle contromisure rozze per iniziare
    # simil sigmoide. 
    # https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_optim/ode_optim.md
    # beta feature
    # NAR = model.node_at_risk(model)
    # model.reduce_migration_rate!(NAR)
    slope(x, β) = 1 / (1 + (x / (1 - x))^(-β))
    length(data[!, :infected_status]) == 0 && return
    # cappo il ratio tra [-1,1]
    ratio = length(data[!, 1]) / (data[end, :infected_status] - data[1, :infected_status])
    s = slope(abs(ratio), β)
    # rapida crescita, lenta decrescita
    if ratio > 0
        model.η = s ≥ model.η ? s : model.η * (1 + s)
        model.η = model.η ≥ 1 ? 1 : model.η
    elseif ratio < 0
        model.η /= (1 + s)
    end
    # if model.ξ == 0 && rand(model.rng) < 1 / 40
    #     model.ξ = abs(rand(Normal(0.0003, 0.00003)))
    # end
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
function predict(data::DataFrame, tspan::Tuple; seed=1337)
    tspan = float.(tspan)
    rng = StableRNG(seed)
    X = Array(data)'
    rbf(x) = exp.(-(x .^ 2))

    # define our UDE. We will use Lux.jl to define the neural network
    # Multilayer FeedForward (5 states S,E,I,R,D and 8 hidden, activation tanh or rbf)
    U = Lux.Chain(Lux.Dense(5, 8, rbf),
        Lux.Dense(8, 8, rbf), Lux.Dense(8, 8, rbf),
        Lux.Dense(8, 5))
    p, st = Lux.setup(rng, U)

    # we define the UDE as a dynamical system
    function ude_dynamics!(du, u, p, t)
        dS, dE, dI, dR, dD = U(u, p, st)[1]
        [dS, dE, dI, dR, dD]
    end
    prob_nn = ODEProblem(ude_dynamics!, X[:, 1], tspan, p)

    # let's build a training loop around our UDE
    # function predict which runs our simulation at new
    # neural network weights. Reacall that weights are the parameters
    # of the ODE, so we want to update the parameters and then run 
    # again
    function predict(θ, X=X[:, 1], T=tspan)
        _prob = remake(prob_nn, u0=X, tspan=(T[1], T[end]), p=θ)
        Array(solve(_prob, Vern7(), saveat=T, abstol=1e-6, reltol=1e-6))
    end

    # for our loss function we solve the ODE at our new parameters
    # and check its L2 loss against the dataset
    function loss(θ)
        X̂ = predict(θ)
        mean(abs2, X .- X̂)
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
    res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=5000)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")
    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=1000)
    println("Final training loss after $(length(losses)) iterations: $(losses[end])")

    # Rename the best candidate
    p_trained = res2.u
    ts = first(tspan):(mean(diff(tspan))/2):last(tspan)
    X̂ = predict(p_trained, X[:, 1], ts)
    Ŷ = U(X̂, p_trained, st)[1]
    return X̂, Ŷ
end

# https://docs.sciml.ai/Overview/stable/showcase/optimization_under_uncertainty/
function policy!(model::StandardABM, data::DataFrame)
    # cerco di massimizzare la happiness e minimizzare gli infetti
end

function policy!(data::DataFrame; seed=1337)
    # https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
    # https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
    rng = Xoshiro(seed)
end

end
