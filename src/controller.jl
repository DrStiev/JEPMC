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

# include("params.jl")
# include("graph.jl")

# TODO RIVEDIMI!
# df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
# date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")
# abm_parameters = model_params.get_abm_parameters(20, 0.01, 3300)
# model = graph.init(; abm_parameters...)

# data = graph.collect(model, graph.agent_step!, graph.model_step!; n = 100)
# X = float.(Array(select(data, [:infected_detected, :controls])))
# # X = float.(Array(select(data, [:susceptible_status, :exposed_status, :infected_status, :dead])))
# ts = (1.0:length(data[!, 1]))

# # con piu' di 2 equazioni scoppia
# sys, params = model_params.system_identification(X', ts)
# println(params)
# println(sys)

function countermeasures!(model::StandardABM, data::DataFrame)
    # applico delle contromisure rozze per iniziare
    length(data[!, :infected_status]) == 0 && return
    start = data[1, :infected_status]
    finish = data[end, :infected_status]
    ratio = (finish - start) / start
    if ratio > 0
        # predict!(data)
        model.η = model.η ≥ 1 ? 1 : model.η * 2
    elseif ratio < 0
        model.η /= 2
    end
    if model.ξ == 0 && rand(model.rng) < 1 / 40
        model.ξ = abs(rand(Normal(0.0003, 0.00003)))
    end
end

function predict(model::StandardABM, data::DataFrame, timeshift::Int)
    # predico l'andamento futuro di tot passi con i dati che ho 
end

# TODO: https://docs.sciml.ai/Overview/stable/showcase/optimization_under_uncertainty/
function policy!(model::StandardABM, data::DataFrame)
    # cerco di massimizzare la happiness e minimizzare gli infetti
end

function policy!(data::DataFrame; seed = 1337)
    # https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
    # https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
    # https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
    # https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
    rng = Xoshiro(seed)
end

end
