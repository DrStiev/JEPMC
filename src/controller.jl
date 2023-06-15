module controller

using Agents, DataFrames
using Statistics: mean

include("utils.jl")
# parametri su cui il controllore può agire:
# η → countermeasures (0.0 - 1.0)
# Rᵢ → objective value for R₀
# ξ → vaccination rate

# https://github.com/epirecipes/sir-julia
function countermeasures!(
    model::StandardABM,
    prediction::Matrix{Float64},
    tshift::Int64,
    step::Float64;
    mininfects = 1,
)

    # control over row 3 and 5 for status :I and :D
    # return the slope of a tanh (- if - and + if +)
    slope(pred) = tanh(pred[3, end] - pred[3, 1]) + tanh(pred[5, end] - pred[5, 1])

    # get the infection rate for each node
    function get_node_status(model::StandardABM, pos::Int)
        agents = filter(x -> x.pos == pos, [a for a in allagents(model)])
        infects = filter(x -> x.status == :I, agents)
        # return length(infects) / length(agents)
        # iff there are a minimum number of infections
        if length(infects) > mininfects
            return length(infects) / length(agents)
        else
            return 0.0
        end
    end

    function update_migration_rates!(model::StandardABM, node::Int, restriction::Float64)
        # update the migration matrix
        model.new_migration_rate = model.migration_rate
        # apply a sort of lockdown
        model.new_migration_rate[node, :] -= model.migration_rate[node, :] * restriction
        model.new_migration_rate[:, node] -= model.migration_rate[:, node] * restriction
        model.new_migration_rate[node, node] +=
            model.migration_rate[node, node] * restriction
        # normalize the matrix between 0 and 1
        model.new_migration_rate[model.new_migration_rate.<0.0] .= 0.0
        model.new_migration_rate[model.new_migration_rate.>1.0] .= 1.0
    end

    rate = slope(prediction[:, (end-(tshift/step)):end])
    # apply countermeasures and update the model
    for i = 1:length(model.η)
        if get_node_status(model, i) > 0.0
            # applico le contromisure solamente se il nodo ha un status > 0
            if rate > 0.0
                model.η[i] = rate ≥ model.η[i] ? rate : model.η[i]
                # apply lockdown only if rate is too high
                # not too sure about this
                if rate > 0.1
                    update_migration_rates!(model, i, rate)
                end
                # balance the countermeasures with a simple formula
                h = model.happiness[i]
                model.η[i] =
                    h + model.η[i] < model.η[i] / 2 ?
                    model.η[i] * (1 - (model.η[i] / abs(h))) : model.η[i]
            elseif rate < 0.0
                model.η[i] *= (1.0 + rate)
            end
        end
    end
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
predict(data::DataFrame, tspan::Int; seed = 1337, maxiters = 5000) =
    udePredict.ude_prediction(data, tspan; seed = seed, maxiters = maxiters)

end
