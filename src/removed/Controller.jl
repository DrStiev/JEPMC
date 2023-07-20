using Agents, DataFrames, Random, Distributions, Distributed
using Statistics: mean

# https://github.com/epirecipes/sir-julia

include("ControllerUtils.jl")

"""
    vaccine!(model=StandardABM, [avg_effectiveness=Float64]; [time=Int])

    Function that is used to simulate the research for a vaccine. This
    research gives then a vaccine with an average effectiveness sample
    from a normal distribution with mean = avg_effectiveness and variance = avg_effectiveness / 10

    The result is a vaccine with total coverage from all the variants that exist in the model
    and with a tolerance from variation that depends on how quick the vaccine is being
    released. Quicker vaccine means a lower tolerance to variation.

# Example
```jldoctest
julia> vaccine!(model)

```
"""
function vaccine!(model::StandardABM, avg_effectiveness::Float64 = 0.83; time::Int = 365)
    if rand(model.rng) < 1 / time
        v =
            (1 - (1 / model.R₀ᵢ)) /
            rand(model.rng, Normal(avg_effectiveness, avg_effectiveness / 10))
        model.ξ = v / model.ω
        model.vaccine_coverage = model.all_variants
        model.variant_tolerance =
            round(Int, tanh(model.step_count / time * avg_effectiveness) * 1024)
    end
end

"""
    slope(pred=Matrix)

    Function that is used to obtain a value for the increase or decrease
    of the infected and dead agents.

# Example
```jldoctest
julia> slope(pred)

```
"""
slope(pred::Matrix) = tanh(pred[3, end] - pred[3, 1]) + tanh(pred[5, end] - pred[5, 1])

"""
    get_node_status(model=StandardABM, pos=Int; [mininfects=Int])

    Function that return the status of a node. If the number of infected individuals in a
    node is less or equal than mininfects, than the node status is 0. Otherwise,
    the node status is infected individual / total individuals

# Example
```jldoctest
julia> get_node_status(model, 1)

```
"""
function get_node_status(model::StandardABM, pos::Int; mininfects::Int = 2)
    agents = filter(x -> x.pos == pos, [a for a in allagents(model)])
    infects = filter(x -> x.status == :I, agents)
    if length(infects) > mininfects
        return length(infects) / length(agents)
    else
        return 0.0
    end
end

"""
    local_controller!(model=StandardABM, data=Matrix, node=Int, step=Int; [mininfects=Int], [vaccine=Bool], [check_happiness=Bool])

    Function that is used to apply a countermeasure to a specific node of a the model
"""
function local_controller!(
    model::StandardABM,
    data::Matrix,
    node::Int,
    step::Int;
    mininfects::Int = 1,
    vaccine::Bool = false,
    check_happiness::Bool = true,
)

    rate = slope(data[:, (end-step)+1:end])
    if rate > 0.0
        model.η[node] = rate ≥ model.η[node] ? rate : model.η[node]
    else
        model.η[node] *= (1.0 + rate)
    end
    if check_happiness
        happiness!(model)
    end
end

"""
    global_controller!(model=StandardABM, ns=Vector, restriction=Vector)

    Function that is used to apply the countermeasures or restriction to the entire model
    changing the migration flux rate of the agents between the nodes, applying a sort
    of lockdown strategy
"""
function global_controller!(
    model::StandardABM,
    ns::Vector{Float64},
    restriction::Vector{Float64},
)
    appo = ns .> 0.0
    for i = 1:length(appo)
        if appo[i] == true
            model.migration_rate[i, :] -= model.migration_rate[i, :] * restriction[i]
            model.migration_rate[:, i] -= model.migration_rate[:, i] * restriction[i]
            model.migration_rate[i, i] += (1 - model.migration_rate[i, i]) * restriction[i]
            model.migration_rate[model.migration_rate.<0.0] .= 0.0
            model.migration_rate[model.migration_rate.>1.0] .= 1.0
        end
    end
end

"""
    happiness!(model=StandardABM)

    Function that estimate in a very rough way the impact of the countermeasures given
    the happiness of a node. This is used to counterbalance the strictness of them to
    avoid to fall into a bad loop of too high and strick countermeasures that will be
    realistically impossible to mantain
"""
function happiness!(model::StandardABM)
    for i = 1:length(model.η)
        h = model.happiness[i]
        model.η[i] =
            h + model.η[i] < model.η[i] / 2 ? model.η[i] * (1 - (model.η[i] / abs(h))) :
            model.η[i]
    end
end

function voc()
    # prova a predire quando uscira' la nuova variante
    # idea molto ambiziosa
end

"""
    predict(model=StandardABM, nodes=Vector{Float64}, tspan=Int; [traindata_size=Int])
"""
function predict(
    model::StandardABM,
    nodes::Vector{Float64},
    tspan::Int;
    traindata_size::Int = 30,
)
    # TODO: predico andamento del nodo più brutto e uso quella predizione per gli altri
    pred = []
    for n = 1:length(nodes)
        if nodes[n] ≠ 0.0
            p_true = [model.R₀, model.γ, model.σ, model.ω, model.δ, model.η[n], model.ξ]
            df = filter(:node => ==(n), model.outresults)
            data = select(df, [:susceptible, :exposed, :infected, :recovered, :dead])
            l = length(data[:, 1])
            # TODO: rivedi logica
            l == 0 && return nothing
            traindata_size = min(l, traindata_size)
            res = nothing
            try
                res = udePredict.ude_prediction(
                    data[:, l-trainingdata_size+1:l],
                    p_true,
                    tspan,
                )
            catch ex
                isdir("data/error/") == false && mkpath("data/error/")
                joinpath("data/error/", "log_" * string(today()) * ".txt")
                log = @error "prediction failed" exception = (ex, catch_backtrace())
                open("data/error/log_" * string(today()) * ".txt", "a") do io
                    write(io, log)
                end
                AgentsIO.save_checkpoint(
                    "data/error/abm_checkpoint_" * string(today()) * ".jld2",
                    model,
                )
                save_dataframe(data, "data/error/", "abm_dataframe")
            finally
                push!(pred, res)
            end
        else
            push!(pred, nothing)
        end
    end
    return pred
end
