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
    step::Float64,
)
    function slope(pred)
        # control over row 3 and 5 for status :I and :D
        # return the slope of a tanh (- if - and + if +)
        return tanh(pred[3, end] - pred[3, 1]) + tanh(pred[5, end] - pred[5, 1])
    end

    # prediction part
    rate = slope(prediction[:, (end-(tshift/step)):end])
    # 3. apply countermeasures and update the model
    if rate > 0.0
        for i = 1:length(model.η)
            model.η[i] = rate ≥ model.η[i] ? rate : model.η[i]
        end
    elseif rate < 0.0
        for i = 1:length(model.η)
            model.η[i] *= (1.0 + rate)
        end
    end
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
function predict(data::DataFrame, tspan::Int; seed = 1337, maxiter = 1000)
    pred, ts = udePredict.ude_prediction(data, tspan; seed = seed, maxIters = maxiter)
    # estimate_long, pred, ts = udePredict.ude_prediction(
    #     data,
    #     tspan;
    #     seed=seed,
    #     maxIters=maxiter
    # )
    # return estimate_long, pred, ts
    return pred, ts
end

end
