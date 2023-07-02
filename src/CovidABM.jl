using Agents
using Graphs
using SparseArrays: findnz
using Random
using LinearAlgebra: diagind
using GraphPlot
using StatsBase: sample, Weights
using Statistics: mean
using Colors

import OrdinaryDiffEq

@agent Node ContinuousAgent{2} begin
    N::Int
    status::Vector{Float64} # S, E, I, R
    happiness::Float64
    η::Float64
    R₀::Float64
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
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0],
    avgPopulation::Int=3300,
    maxTravelingRate::Float64=0.1,
    spacing::Float64=4.0,
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
            :population => population,
            :migrationMatrix => migrationMatrix,
        ),
        rng
    )

    Is = [zeros(Int, numNodes)...]
    for i = 1:initialNodeInfected
        Is[rand(model.rng, 1:numNodes)] = 1
    end

    for node in 1:numNodes
        # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
        r = model.space.extent[1] * 0.5 * sqrt(rand(model.rng))
        θ = 2 * pi * rand(model.rng)
        position = model.space.extent .* 0.5 .+ (r * cos(θ), r * sin(θ)) .- 0.5
        status = Is[node] == 1 ? [(population[node] - 1) / population[node], 0.0, 1 / population[node], 0.0] : [1.0, 0.0, 0.0, 0.0]
        happiness = randn()
        happiness = happiness < -1.0 ? -1.0 : happiness > 1.0 ? 1.0 : happiness
        add_agent!(position, model, (0, 0), population[node], status, happiness, 0.0, param[1])
    end
    return model
end

model = init()

# utile per plot
max = maximum(model.population)
status = [a.status for a in allagents(model)]
nodefillc = [RGBA(1.0 * (status[i][2] + status[i][3]), 1.0 * status[i][1], 1.0 * status[i][4], 1.0) for i in 1:length(status)]
gplot(model.connections, nodesize=model.population ./ max, nodefillc=nodefillc)

function model_step!(model::ABM)
    update!(model)
    happiness!(model)
    voc!(model)
    controller!(model)
end

# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/
function update!(model::ABM)
    # faccio una chiamata all'integrator con step ≈ 14gg
    OrdinaryDiffEq.step!(model.i, model.param[2]^-1, true)
    # notifico l'integrator che i dati sono stati aggiornati
    OrdinaryDiffEq.u_modified!(model.i, true)
    OrdinaryDiffEq.step!(model.i, 1.0, true)

end

function happiness!(model::ABM)
end

function voc!(model::ABM)
end

function controller!(model::ABM)
end

function agent_step!(agent, model::ABM)
    migrate!(agent, model)
end

function migrate!(agent, model::ABM)
    network = model.migrationMatrix[agent.id, :]
    tidxs, tweights = findnz(network)

    for i in 1:length(tidxs)
        people_traveling_out = round(Int, agent.N * tweights[i])
        people_status = map((x) -> round(Int, x), agent.status .* people_traveling_out)

        for j in length(people_status)
            objective = filter(x -> x.id == tidxs[i], [a for a in allagents(model)])
            objective.status = objective.status .* objective.N
            objective.status[j] += people_status[j]
            objective.N += people_status[j]
            objective.status = objective.status ./ objective.N

            agent.status = agent.status .* agent.N
            agent.status[j] -= agent.status[j]
            agent.N -= agent.status[j]
            agent.status = agent.status ./ agent.N
        end
    end
end
