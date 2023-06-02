using BlackBoxOptim, Random, Agents
using Statistics: mean

include("utils.jl")
include("graph.jl")

function cost(x, step)
    infected_fraction(model) =
        count(a.status == :I for a in allagents(model)) / nagents(model)

    res = Float64[]

    callback = function (l)
        push!(res, l)
        display(res)
    end

    for _ = 1:step
        model = graph.init(; x...)
        _, mdf = run!(
            model,
            graph.agent_step!,
            graph.model_step!,
            50;
            mdata = [infected_fraction],
            when_model = [50],
            showprogress = true,
        )
        callback(mdf.infected_fraction[1])
    end
    return mean(res)
end

x = parameters.get_abm_parameters(20, 0.01, 3300)
cost(x, 10)

result = bboptimize(
    cost,
    SearchRange = [
        x[:R₀],
        x[:γ],
        x[:σ],
        x[:ξ],
        x[:ω],
        x[:migration_rate],
        x[:δ],
        x[:number_point_of_interest],
        x[:Rᵢ],
        (0.0, 1.0),
    ],
    NumDimensions = length(x),
    MaxTime = 20,
)
