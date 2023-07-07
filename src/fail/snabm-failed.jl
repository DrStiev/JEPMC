using Agents, Graphs, Random, GraphPlot, Colors
using DiffEqCallbacks, Distributions, Plots, DataFrames
using SparseArrays: findnz
using LinearAlgebra: diagind
using StatsBase: sample, Weights
using Statistics: mean

import OrdinaryDiffEq, DiffEqCallbacks

@agent Node ContinuousAgent{2} begin
    population::Float64
    status::Vector{Float64} # S, E, I, R, D
    param::Vector{Float64} # R₀, γ, σ, ω, δ, η, ξ
    happiness::Float64
end

adapt_R₀!(x) = return 1.1730158534328545 + 0.21570538523224972 * x

function get_migration_matrix(g::SimpleGraph, population::Vector{Int}, numNodes::Int, maxTravelingRate::Float64)
    migrationMatrix = zeros(numNodes, numNodes)

    for n = 1:numNodes
        for m = 1:numNodes
            migrationMatrix[n, m] = (population[n] + population[m]) / population[n]
        end
    end

    migrationMatrix = (migrationMatrix .* maxTravelingRate) ./ maximum(migrationMatrix)
    migrationMatrix[diagind(migrationMatrix)] .= 1.0
    mmSum = sum(migrationMatrix, dims=2)

    for c = 1:numNodes
        migrationMatrix[c, :] ./= mmSum[c]
    end

    return migrationMatrix .* adjacency_matrix(g)
end

function generate_nearly_complete_graph(n::Int, N::Int; seed::Int=1234)
    (n*(n-1)/2) - N < n && throw("The number of edges that will be removed [$N] will prevent the construction of a connected graph of [$n] nodes!")
    rng = Xoshiro(seed)
    g = complete_graph(n)

    # Remove N random edges
    e = Graphs.collect(Graphs.edges(g))
    shuffled_edges = e[randperm(rng, length(e))]
    edges_to_remove = shuffled_edges[1:N]
    for e in edges_to_remove
        rem_edge!(g, e)
    end

    return g
end
nv()
# https://juliadynamics.github.io/Agents.jl/stable/examples/schoolyard/
function init(;
    numNodes::Int=50,
    edgesCoverage::Float64=0.6,
    initialNodeInfected::Int=1,
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007],
    avgPopulation::Int=3300,
    maxTravelingRate::Float64=0.1,  # flusso di persone che si spostano
    tspan::Tuple=(1.0, Inf),
    seed::Int=42
)

    rng = Xoshiro(seed)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    graph = generate_nearly_complete_graph(numNodes, floor(Int, (1 - edgesCoverage) * (numNodes * (numNodes - 1) / 2)); seed=seed)
    migrationMatrix = get_migration_matrix(graph, population, numNodes, maxTravelingRate)

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

    function seir!(du, u, p, t)
        S, E, I, R, D = u
        R₀, γ, σ, ω, δ, η, ξ = p
        du[1] = (-R₀ * γ * S * I) + (ω * R) - (S * ξ) # dS
        du[2] = (R₀ * γ * S * I) - (σ * E) # dE
        du[3] = (σ * E) - (γ * I) - (δ * I) # dI
        du[4] = ((1 - δ) * γ * I - ω * R) + (S * ξ) # dR
        du[5] = (δ * I * γ) # dD
    end

    prob = [OrdinaryDiffEq.ODEProblem(seir!, a.status, tspan, a.param) for a in allagents(model)]
    integrator = [OrdinaryDiffEq.init(p, OrdinaryDiffEq.Tsit5(); advance_to_tstop=true) for p in prob]
    model.properties[:integrator] = integrator

    return model
end

function model_step!(model::ABM)
    agents = [agent for agent in allagents(model)]
    for agent in 1:length(agents)
        # notify the integrator that the condition may be altered
        OrdinaryDiffEq.u_modified!(model.integrator[agent], true)
        OrdinaryDiffEq.step!(model.integrator[agent], 1.0, true)
        OrdinaryDiffEq.u_modified!(model.integrator[agent], true)
        agents[agent].status = model.integrator[agent].u
    end
    voc!(model)
    # controller!(model)
end

function agent_step!(agent, model::ABM)
    migrate!(agent, model)
    dead!(agent)
    update!(agent)
    happiness!(agent)
end

function dead!(agent)
    agent.population -= round(Int, agent.status[5] * agent.population)
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
        agent.population -= sum(people_traveling_out)
        agent.status ./= agent.population
        agent.population = map((x) -> round(Int, x), agent.population)

        objective = filter(x -> x.id == tidxs[i], [a for a in allagents(model)])[1]
        objective.status = (objective.status .* objective.population) .+ people_traveling_out
        objective.population += sum(people_traveling_out)
        objective.status ./= objective.population
        objective.population = map((x) -> round(Int, x), objective.population)
    end
end

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

function controller!(model::ABM)
end

function plot_system(model::ABM)
    # utile per plot
    max = maximum([sum(agent.status) for agent in allagents(model)])
    status = [a.status ./ sum(a.status) for a in allagents(model)]
    nodefillc = [RGBA(1.0 * (status[i][2] + status[i][3]), 1.0 * status[i][1], 1.0 * status[i][4], 1.0) for i in 1:length(status)]
    gplot(model.connections, nodesize=[sum(agent.status) for agent in allagents(model)] ./ max, nodefillc=nodefillc, nodelabel=sort([agent.id for agent in allagents(model)]))
end

function get_observable_data()
    status(x) = x.status
    happiness(x) = x.happiness
    η(x) = x.param[6]
    R₀(x) = x.param[1]
    return [status, happiness, η, R₀]
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

# TODO: il migration matrix influisce sensibilmente sul tipo di grafico che ottengo del modello
# TODO: modello sensibile anche al seed e alla randomicita'
model = init(; numNodes=4, maxTravelingRate=0.1, edgesCoverage=0.6)

models = [init(numNodes=10, maxTravelingRate=0.01, seed=abs(i)) for i in rand(Int64, 10)]
data = ensemble_collect(models)

plot_system(model)
model.migrationMatrix

data = collect(model; n=10)
plot_system(model)

plot(Array(DataFrame(Array(ares[!, :status]), :auto))', label=["S" "E" "I" "R" "D"],)
mres

# TODO: fare grafico decente
# TODO: applicare controller. lockdown viene gestito con migrationrate e ricalcolo migration maxMatrix
