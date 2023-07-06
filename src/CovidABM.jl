using Agents, Graphs, Random, GraphPlot, Colors
using DiffEqCallbacks, Distributions, Plots
using SparseArrays: findnz
using LinearAlgebra: diagind
using StatsBase: sample, Weights
using Statistics: mean

import OrdinaryDiffEq

@agent Node ContinuousAgent{2} begin
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

    maxMatrix = maximum(migrationMatrix)
    migrationMatrix = (migrationMatrix .* maxTravelingRate) ./ maxMatrix
    migrationMatrix[diagind(migrationMatrix)] .= 1.0
    mmSum = sum(migrationMatrix, dims=2)

    for c = 1:numNodes
        migrationMatrix[c, :] ./= mmSum[c]
    end

    return migrationMatrix .* adjacency_matrix(g)
end

function generate_random_connected_graph(n::Int; seed::Int=1234)
    rng = Xoshiro(seed)
    g = SimpleGraph(n)

    for v in 2:n
        u = rand(rng, 1:v-1)
        add_edge!(g, u, v)
    end

    return g
end

# https://juliadynamics.github.io/Agents.jl/stable/examples/schoolyard/
function init(;
    numNodes::Int=50,
    initialNodeInfected::Int=1,
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007],
    avgPopulation::Int=3300,
    maxTravelingRate::Float64=0.7,  # flusso di persone che si spostano
    spacing::Float64=4.0,
    tspan::Tuple=(1.0, Inf),
    seed::Int=1234
)

    rng = Xoshiro(seed)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    graph = generate_random_connected_graph(numNodes; seed=seed)
    migrationMatrix = get_migration_matrix(graph, population, numNodes, maxTravelingRate)

    model = ABM(
        Node,
        ContinuousSpace((100, 100); spacing=spacing, periodic=true);
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
        r = model.space.extent[1] * 0.25 * sqrt(rand(model.rng))
        θ = 2 * pi * rand(model.rng)
        position = model.space.extent .* 0.5 .+ (r * cos(θ), r * sin(θ)) .- 0.5
        status = Is[node] == 1 ? [population[node] - 1, 0, 1, 0, 0] : [population[node], 0, 0, 0, 0]
        happiness = randn(model.rng)
        happiness = happiness < -1.0 ? -1.0 : happiness > 1.0 ? 1.0 : happiness
        parameters = vcat(param, [0.0, 0.0])
        add_agent!(position, model, (0, 0), status, parameters, happiness)
    end

    function seir!(du, u, p, t)
        S, E, I, R, D = u
        N = S + E + I + R
        R₀, γ, σ, ω, δ, η, ξ = p
        du[1] = ((-R₀ * γ * S * I) / N) + (ω * R) - (S * ξ) # dS
        du[2] = ((R₀ * γ * S * I) / N) - (σ * E) # dE
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
        migrate!(agents[agent], model, agent)
        OrdinaryDiffEq.step!(model.integrator[agent], 1.0, true)
        OrdinaryDiffEq.u_modified!(model.integrator[agent], true)
        agents[agent].status = model.integrator[agent].u
        OrdinaryDiffEq.u_modified!(model.integrator[agent], true)
        update!(agents[agent])
        happiness!(agents[agent])
    end
    voc!(model)
    # controller!(model)
end

# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/
function update!(agent)
    if agent.param[1] > 1.0
        agent.param[1] -= agent.param[6] * (agent.param[1] - 1.0)
        index = findfirst(a -> a.id == agent.id, [a for a in allagents(model)])
        OrdinaryDiffEq.u_modified!(model.integrator[index], true)
    end
end

function migrate!(agent, model::ABM, index::Int)
    network = model.migrationMatrix[agent.id, :]
    tidxs, tweights = findnz(network)

    for i in 1:length(tidxs)
        people_traveling_out = map((x) -> round(Int, x), agent.status .* tweights[i])
        people_traveling_out[end] = 0

        objective = filter(x -> x.id == tidxs[i], [a for a in allagents(model)])[1]
        objective.status += people_traveling_out
        agent.status -= people_traveling_out

        model.integrator[index].u = agent.status
        OrdinaryDiffEq.u_modified!(model.integrator[index], true)
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

            index = findfirst(a -> a.id == agent.id, [a for a in allagents(model)])
            OrdinaryDiffEq.u_modified!(model.integrator[index], true)
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
    gplot(model.connections, nodesize=[sum(agent.status) for agent in allagents(model)] ./ max, nodefillc=nodefillc, nodelabel=[agent.id for agent in allagents(model)])
end

function get_observable_data()
    status(x) = x.status
    happiness(x) = x.happiness
    η(x) = x.param[6]
    R₀(x) = x.param[1]
    return [status, happiness, η, R₀]
end

model = init(numNodes=4, maxTravelingRate=0.7)
initial = sum([sum(agent.status) for agent in allagents(model)])
plot_system(model)

model.migrationMatrix

ares, mres = run!(model, dummystep, model_step!, 1200; showprogress=true, adata=get_observable_data())
ending = sum([sum(agent.status) for agent in allagents(model)])
plot_system(model)
initial - ending == 0.0 # dovrebbe essere 0
ares
mres
using DataFrames
res = [filter(:id => ==(i), ares) for i in unique(ares[!, :id])]
plt = []

for r in res
    x = DataFrame(Array(r[!, :status]), :auto)
    y = DataFrame(Array(select(r, [:happiness, :η])), :auto)
    z = DataFrame(Array(select(r, [:R₀])), :auto)
    push!(
        plt,
        plot(
            plot(Array(x)', label=["S" "E" "I" "R" "D"]),
            plot(Array(y), label=["Happiness" "η"]),
            plot(Array(z), label="R₀"),
        )
    )
end
plot(plt...)

# TODO: fare grafico decente, con tanto di grafo
# TODO: applicare controller. lockdown viene gestito con migrationrate e ricalcolo migration maxMatrix
# TODO: il migration matrix influisce sensibilmente sul tipo di grafico che ottengo del modello
