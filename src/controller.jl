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
function countermeasures!(model::StandardABM, data::DataFrame, tspan::Int; β=3, saveat=3, seed = 1337)
    # applico delle contromisure rozze per iniziare
    # https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_optim/ode_optim.md
    lestimate, x, y = predict(data, tspan; seed=seed)
end

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
function predict(data::DataFrame, tspan::Int; seed=1337)
    X, Y = udePredict.ude_prediction(data, tspan; seed=seed)
    long_estimate = udePredict.symbolic_regression(X, Y, tspan; seed=seed)
    return long_estimate, X, Y
end

end
