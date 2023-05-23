module controller
# for the neural network training
using OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
# for the symbolic model discovery
using ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse, Zygote
# external libraries
using Lux, ComponentArrays, DataFrames, Plots, Random, Agents, Distributions, LinearAlgebra, Statistics
using DrWatson: @dict

# include("graph.jl")

# parametri su cui il controllore può agire:
# ξ → percentage of population vaccined per model step [0.0 - 0.03]
# η → countermeasures [1.0 - 0.0)
# θ → percentage of people under generalized lockdown [0.0 - 1.0).
# θₜ → total number of days (model step) in which θ is applied
# q → days of quarantine for each infected agent detected [0.0 - γ*2]
# ncontrols → percentage of controls per day

# include("params.jl")

# df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
# date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")
# abm_parameters = model_params.get_abm_parameters(20, 0.01, 3300)
# model = graph.init(; abm_parameters...)

# data = graph.collect(model, graph.agent_step!, graph.model_step!; n=21)
# X = float.(Array(select(data, [:infected_detected, :controls])))
# # X₀ = float.(Array(select(data, [:susceptible_status, :exposed_status, :infected_status, :dead])))
# ts = (1.0:length(data[!, 1]))

# rbf(x) = exp.(-(x .^ 2))
# rng = Xoshiro(1111)
# # Multilayer FeedForward
# U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
#     Lux.Dense(5, 2))
# # Get the initial parameters and state variables of the model
# p, st = Lux.setup(rng, U)

# # con piu' di 2 equazioni scoppia
# sys, params = model_params.system_identification(X', ts)
# println(params)
# println(sys)

function controls!(data::DataFrame)
    # check the slope of the line to see the growth of the pandemic
    # adjust controls by a factor similar to the growth of the pandemic
    i₀ = data[1, :infected_detected]
    iₙ = data[end, :infected_detected]
    m = (iₙ - i₀) / i₀
    ncontrols = data[end, :controls]
    ncontrols += (ncontrols * m)
    properties[:trend] = m
    properties[:controls] = ncontrols
end

function countermeasures!(data::DataFrame)
    # applicare contromisure in base ai controlli e
    # general trend della curva epidemica
    if properties[:trend] > 0
        if properties[:trend] < 0.5
            properties[:countermeasures_speed] = 1 / 20
        elseif properties[:trend] < 1.0
            properties[:countermeasures_speed] = 1 / 5
        else
            if properties[:lockdown_time] == 0
                properties[:lockdown_time] = abs(rand(Normal(60, 30)))
                properties[:lockdown_percentage] = abs(rand(Normal(0.3, 0.3)))
            end
        end
    else
        if properties[:countermeasures_speed] > 0.0
            properties[:countermeasures_speed] = 1 / 20
        end
    end
end

function policy!(data::DataFrame, properties, minimize, maximize; saveat=1, seed=1337)
    # https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
    # https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
    # https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
    # https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

    # TODO: gestire le properties da ritornare, i 
    # casi con valori NaN e Inf e tutto l'ambaradam.
    # trovare un modo per far funzionare tutto

    rng = Xoshiro(seed)
    controls!(data)
    countermeasures!(data)
    return properties
end

properties = @dict(
    controls = 0,
    quarantine_time = 0,
    lockdown_time = 0,
    lockdown_percentage = 0.0,
    vaccine_per_day_percentage = 0.0,
    countermeasures_speed = 0.0,
    trend = 0.0,
)
end
