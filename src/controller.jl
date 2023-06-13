module controller

using Agents, DataFrames
include("utils.jl")
# parametri su cui il controllore può agire:
# η → countermeasures (0.0 - 1.0)
# Rᵢ → objective value for R₀
# ξ → vaccination rate

# https://github.com/epirecipes/sir-julia
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_optimization/ode_lockdown_optimization.md
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ude/ude.md
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_ddeq/ode_ddeq.md
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_optim/ode_optim.md
function countermeasures!(
    model::StandardABM,
    data::DataFrame,
    tspan::Int;
    seed = 1337,
    maxiter = 1000,
)
    function slope(pred, step)
        # control over row 3 and 5 for status :I and :D
        return (length(pred[1, :]) * step) / (pred[3, end] - pred[3, 1])
    end

    function balance_happiness(model::StandardABM)
        # balance happiness predicting how it will be affected
        # by the use of the actual countermeasures
    end

    total = sum(data[1, :])
    # 1. prediction with actual countermeasures over period of time
    lestimate, pred, ts = predict(data, tspan; seed = seed, maxiter = maxiter)
    # 2. prediction of the countermeasures
    l = length(data[!, 1])
    step = ts[2] - ts[1] # time interval
    p = Array(pred[1] .* total)[:, l/step:end] # prediction part
    rate = slope(p, step)
    # 3. apply countermeasures and update the model
    if rate > 0.0
        model.η = rate ≥ model.η ? tanh(rate) : model.η
    elseif rate < 0.0
        model.η /= (1.0 + tanh(rate))
    end
    balance_happiness(model)
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
function predict(data::DataFrame, tspan::Int; seed = 1337, maxiter = 1000)
    pred, ts = udePredict.ude_prediction(data, tspan; seed = seed, maxIters = maxiter)
    long_estimate = udePredict.symbolic_regression(pred[1], pred[2], tspan; seed = seed)
    return long_estimate, pred, ts
end

end
