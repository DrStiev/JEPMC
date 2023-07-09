using Agents, Graphs, Random, Distributions, DataFrames
using SparseArrays: findnz
using StatsBase: sample, Weights

import OrdinaryDiffEq, DiffEqCallbacks

include("ABMUtils.jl")

@agent Node ContinuousAgent{2} begin
    population::Float64
    status::Vector{Float64} # S, E, I, R, D
    param::Vector{Float64} # R₀, γ, σ, ω, δ, η, ξ
    happiness::Float64
end

# https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_ftc_jump/function_map_ftc_jump.md
# https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_vaccine_jump/function_map_vaccine_jump.md
# questo esempio potrebbe essere buono per la NeuralODE
# https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_optimization/ode_lockdown_optimization.md

# https://juliadynamics.github.io/Agents.jl/stable/examples/schoolyard/
function init(;
    numNodes::Int=20,
    edgesCoverage::Symbol=:high,
    initialNodeInfected::Int=1,
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007],
    avgPopulation::Int=round(Int, 2.9555e6),
    maxTravelingRate::Float64=0.1,  # flusso di persone che si spostano
    tspan::Tuple=(1.0, Inf),
    seed::Int=42
)

    rng = Xoshiro(seed)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    graph = connected_graph(numNodes, edgesCoverage; rng=rng)
    migrationMatrix = get_migration_matrix(graph, population, maxTravelingRate)

    model = ABM(
        Node,
        ContinuousSpace((100, 100); spacing=4.0, periodic=true);
        properties=Dict(
            :numNodes => numNodes,
            :param => param,
            :connections => graph,
            :migrationMatrix => migrationMatrix,
        ),
        rng
    )

    Is = [zeros(Int, numNodes)...]
    for i = 1:initialNodeInfected
        Is[rand(model.rng, 1:numNodes)] = 1
    end

    for node in 1:numNodes
        status = Is[node] == 1 ? [(population[node] - 1) / population[node], 0, 1 / population[node], 0, 0] : [1.0, 0, 0, 0, 0]
        happiness = randn(model.rng)
        happiness = happiness < -1.0 ? -1.0 : happiness > 1.0 ? 1.0 : happiness
        parameters = vcat(param, [0.0, 0.0])
        add_agent!(model, (0, 0), population[node], status, parameters, happiness)
    end

    prob = [OrdinaryDiffEq.ODEProblem(seir!, a.status, tspan, a.param) for a in allagents(model)]
    integrator = [OrdinaryDiffEq.init(p, OrdinaryDiffEq.Tsit5(); advance_to_tstop=true) for p in prob]
    model.properties[:integrator] = integrator

    return model
end

function model_step!(model::ABM)
    agents = [agent for agent in allagents(model)]
    for agent in agents
        # notify the integrator that the condition may be altered
        model.integrator[agent.id].u = agent.status
        model.integrator[agent.id].p = agent.param
        OrdinaryDiffEq.u_modified!(model.integrator[agent.id], true)
        OrdinaryDiffEq.step!(model.integrator[agent.id], 1.0, true)
        agent.status = model.integrator[agent.id].u
    end
    voc!(model)
    controller!(model)
end

function agent_step!(agent, model::ABM)
    migrate!(agent, model)
    update!(agent)
    happiness!(agent)
end

# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/
function update!(agent)
    if agent.param[1] > 1.0
        agent.param[1] -= agent.param[6] * (agent.param[1] - 1.0)
    end
end

# TODO: capire dove perdo individui
function migrate!(agent, model::ABM)
    network = model.migrationMatrix[agent.id, :]
    tidxs, tweights = findnz(network)

    for i in 1:length(tidxs)
        people_traveling_out = agent.status .* tweights[i] .* agent.population
        people_traveling_out[end] = 0

        new_population = agent.population - sum(people_traveling_out)
        agent.status = (agent.status .* agent.population - people_traveling_out) ./ new_population
        agent.population = new_population
        agent.population = map((x) -> round(Int, x), agent.population)

        objective = filter(x -> x.id == tidxs[i], [a for a in allagents(model)])[1]
        new_population = objective.population + sum(people_traveling_out)
        objective.status = (objective.status .* objective.population + people_traveling_out) ./ new_population
        objective.population = new_population
        objective.population = map((x) -> round(Int, x), objective.population)
        indexdst = indexin(objective.id, [a.id for a in allagents(model)])[1]
    end
    indexsrc = indexin(agent.id, [a.id for a in allagents(model)])[1]
end

# TODO: vedere se mantenere campo happiness nel caso, migliorare stimatore
function happiness!(agent)
    agent.happiness = tanh(agent.happiness - agent.param[6])
end

function voc!(model::ABM)
    if rand(model.rng) ≤ 8e-3
        agent = random_agent(model)
        if agent.status[3] ≠ 0.0
            agent.param[1] = rand(model.rng, Uniform(3.3, 5.7))
            agent.param[2] = rand(model.rng, Normal(agent.param[2], agent.param[2] / 10))
            agent.param[3] = rand(model.rng, Normal(agent.param[3], agent.param[3] / 10))
            agent.param[4] = rand(model.rng, Normal(agent.param[4], agent.param[4] / 10))
            agent.param[5] = rand(model.rng, Normal(agent.param[5], agent.param[5] / 10))
        end
    end
end

# TODO: implementare controller
function controller!(model::ABM)
end

function collect(
    model::ABM;
    agent_step=agent_step!,
    model_step=model_step!,
    n=1200,
    showprogress=true,
    split_result=true,
    adata=get_observable_data()
)
    data, _ = run!(
        model,
        agent_step,
        model_step,
        n;
        showprogress=showprogress,
        adata=adata
    )
    if split_result
        return [filter(:id => ==(i), data) for i in unique(data[!, :id])]
    else
        return data
    end
end

function ensemble_collect(
    models::Vector;
    agent_step=agent_step!,
    model_step=model_step!,
    n=1200,
    showprogress=true,
    parallel=true,
    adata=get_observable_data(),
    split_result=true
)
    data, _ = ensemblerun!(
        models,
        agent_step,
        model_step,
        n;
        showprogress=showprogress,
        adata=adata,
        parallel=parallel
    )
    if split_result
        res = [filter(:ensemble => ==(i), data) for i in unique(data[!, :ensemble])]
        outres = []
        for r in res
            r1 = [filter(:id => ==(i), r) for i in unique(r[!, :id])]
            push!(outres, r1)
        end
        return outres
    else
        return data
    end
end
