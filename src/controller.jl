module controller

using Agents, DataFrames, Random, Distributions, Distributed
using Statistics: mean

# https://github.com/epirecipes/sir-julia

@everywhere include("utils.jl")

# TODO: inserire controller in modello
# e attivare sse esplicitato

function controller_vaccine!(model::StandardABM, avg_effectiveness::Float64; time=365)
    if rand(model.rng) < 1 / time
        v =
            (1 - (1 / model.R₀ᵢ)) /
            rand(model.rng, Normal(avg_effectiveness, avg_effectiveness / 10))
        model.ξ = v / model.ω
        model.vaccine_coverage = model.all_variants
        model.variant_tolerance = round(Int, tanh(model.step / time * avg_effectiveness) * 1024)
    end
end

function controller_η!(model::StandardABM, data::Matrix, step::Int; mininfects=1)
    slope(pred) = tanh(pred[3, end] - pred[3, 1]) + tanh(pred[5, end] - pred[5, 1])

    function get_node_status(model::StandardABM, pos::Int)
        agents = filter(x -> x.pos == pos, [a for a in allagents(model)])
        infects = filter(x -> x.status == :I, agents)
        if length(infects) > mininfects
            return length(infects) / length(agents)
        else
            return 0.0
        end
    end

    function apply_lockdown!(model::StandardABM, node::Int, restriction::Float64)
        model.new_migration_rate = model.migration_rate
        model.new_migration_rate[node, :] -= model.migration_rate[node, :] * restriction
        model.new_migration_rate[:, node] -= model.migration_rate[:, node] * restriction
        model.new_migration_rate[node, node] +=
            (1 - model.migration_rate[node, node]) * restriction
        model.new_migration_rate[model.new_migration_rate.<0.0] .= 0.0
        model.new_migration_rate[model.new_migration_rate.>1.0] .= 1.0
    end

    rate = slope(data[:, (end-step)+1:end])
    for i = 1:length(model.η)
        if get_node_status(model, i) > 0.0
            if rate > 0.0
                model.η[i] = rate ≥ model.η[i] ? rate : model.η[i]
                if rate > 0.2
                    apply_lockdown!(model, i, rate)
                end
            else
                model.η[i] *= (1.0 + rate)
            end
        end
    end
end

function controller_happiness!(model::StandardABM)
    for i = 1:length(model.η)
        h = model.happiness[i]
        model.η[i] =
            h + model.η[i] < model.η[i] / 2 ? model.η[i] * (1 - (model.η[i] / abs(h))) :
            model.η[i]
    end
end

function controller_voc()
    # prova a predire quando uscira' la nuova variante
    # idea molto ambiziosa
end
end
