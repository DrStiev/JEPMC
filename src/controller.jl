module controller
# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs

using DataFrames, Plots, Random, Agents, Distributions, LinearAlgebra, Statistics
using DrWatson: @dict

# parametri su cui il controllore può agire:
# η → countermeasures (0.0 - 1.0)
# Rᵢ → objective value for R₀
# ξ → vaccination rate

# TODO RIVEDIMI!
# https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
function sys_id()
    include("params.jl")
    include("graph.jl")

    df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
    date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")
    abm_parameters = model_params.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)

    data = graph.collect(model; n=30, controller_step=7)
    X = Array(select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    ))
    # X = float.(Array(select(data, [:susceptible_status, :exposed_status, :infected_status, :dead])))
    ts = 1:length(data[!, 1])

    # con piu' di 2 equazioni scoppia
    sys, params = model_params.system_identification(X', ts)
    println(params)
    println(sys)
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
function embeddedML()
    # TODO: to be finished
    df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
    date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")

    abm_parameters = model_params.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)

    data = graph.collect(model; n=30, controller_step=7)
    p1 = select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    )
    plot(Array(p1), labels=["S" "E" "I" "R" "D"])

    rng = StableRNG(1111)

    X = Array(select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    ))
    t = length(data[!, 1])

    rbf(x) = exp.(-(x .^ 2))

    # Multilayer FeedForward
    U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
        Lux.Dense(5, 2))
    # Get the initial parameters and state variables of the model
    p, st = Lux.setup(rng, U)
end

function ude_dynamics!(du, u, p, t, p_true)
    # TODO: need to be finished
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ = p_true
    û = U(u, p, st)[1]
    du[1] = -R₀ * γ * S * I
    du[2]
end

function countermeasures!(model::StandardABM, data::DataFrame; β=3)
    slope(x, β) = 1 / (1 + (x / (1 - x))^(-β)) # simil sigmoide
    # applico delle contromisure rozze per iniziare
    length(data[!, :infected_status]) == 0 && return
    # cappo il ratio tra [-1,1]
    ratio = length(data[!, 1]) / (data[end, :infected_status] - data[1, :infected_status])
    s = slope(abs(ratio), β)
    # rapida crescita, lenta decrescita
    if ratio > 0
        if s > model.η
            model.η = s
        else
            model.η *= (1 + s)
        end
        model.η = model.η ≥ 1 ? 1 : model.η
    elseif ratio < 0
        model.η /= (1 + s)
    end
    if model.ξ == 0 && rand(model.rng) < 1 / 40
        model.ξ = abs(rand(Normal(0.0003, 0.00003)))
    end
    return ratio
end

function predict(model::StandardABM, data::DataFrame, tspan)
    # predico l'andamento futuro di tot passi con i dati che ho 
    # e tento di applicare una contromisura adeguata. 
    # se dopo tot passi le mie contromisure hanno 
    # portato una diminuzione nella mia variabile target
    # allora faccio un allentamento, altrimenti aumento
end

# TODO: https://docs.sciml.ai/Overview/stable/showcase/optimization_under_uncertainty/
function policy!(model::StandardABM, data::DataFrame)
    # cerco di massimizzare la happiness e minimizzare gli infetti
end

function policy!(data::DataFrame; seed=1337)
    # https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
    # https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
    rng = Xoshiro(seed)
end

end
