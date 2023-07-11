using Graphs, Random, Agents, DataFrames, Dates, CSV
using Distributions, GraphPlot, Colors, GraphRecipes
using LinearAlgebra: diagind
using Statistics: mean

adapt_R₀!(x) = return 1.1730158534328545 + 0.21570538523224972 * x

function get_migration_matrix(g::SimpleGraph, population::Vector{Int}, maxTravelingRate::Float64)
    numNodes = Graphs.nv(g)
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

function connected_graph(n::Int, coverage::Symbol; rng::AbstractRNG)
    function edge_to_add(n::Int, coverage::Symbol, rng::AbstractRNG)
        low = n - 1
        max = n * (n - 1) / 2
        avg = (n * (n - 1) / 2 + (n - 1)) / 2
        if coverage == :low
            return trunc(Int, rand(rng, low:floor(Int, (avg + low) / 2)) - low)
        elseif coverage == :medium
            return trunc(Int, rand(rng, ceil(Int, (avg + low) / 2):floor(Int, (avg + max) / 2)) - low)
        elseif coverage == :high
            return trunc(Int, rand(rng, ceil(Int, (avg + max) / 2):max) - low)
        end
    end

    function add_random_edges!(graph::SimpleGraph, n::Int; rng::AbstractRNG)
        for i = 1:n
            u = rand(rng, 1:Graphs.nv(graph))
            v = rand(rng, 1:Graphs.nv(graph))
            if u ≠ v
                add_edge!(graph, u, v)
            end
        end
    end

    g = SimpleGraph(n)
    # Create a tree by adding (N-1) edges
    for v in 2:n
        add_edge!(g, v, rand(rng, 1:v-1))
    end
    add_random_edges!(g, edge_to_add(n, coverage, rng); rng=rng)

    return g
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

function plot_system_graph(model::ABM)
    # utile per plot
    max = maximum([sum(agent.population) for agent in allagents(model)])
    status = [a.status for a in allagents(model)]
    nodefillc = [RGBA(1.0 * (status[i][2] + status[i][3]), 1.0 * status[i][1], 1.0 * status[i][4], 1.0) for i in 1:length(status)]
    gplot(model.connections, nodesize=[agent.population for agent in allagents(model)] ./ max, nodefillc=nodefillc, nodelabel=sort([agent.id for agent in allagents(model)]))
end

function get_observable_data()
    status(x) = x.status
    happiness(x) = x.happiness
    η(x) = x.param[6]
    R₀(x) = x.param[1]
    return [status, happiness, η, R₀]
end

function plot_model(data::Vector{DataFrame}; cumulative::Bool=false)
    stateslabel = ["S" " E" "I" "R" "D"]
    cmlabel = ["happiness" "η"]
    rlabel = "R₀"
    plt = []

    l = @layout [
        RecipesBase.grid(1, 1)
        RecipesBase.grid(1, 2)
    ]

    if cumulative
        avg_data = mean([Array(d) for d in data])
        states, cm, r = split_dataset(avg_data)
        plt = plot(
            plot(states', label=stateslabel),
            plot(cm', label=cmlabel),
            plot(r', label=rlabel),
            layout=l
        )
        return plt
    end

    # TODO: sistemare un pochino
    for d in data
        states, cm, r = split_dataset(d)
        push!(plt,
            plot(
                plot(states', label=stateslabel),
                plot(cm', label=cmlabel),
                plot(r', label=rlabel),
                # layout=l
            )
        )
    end
    return plt
end

function split_dataset(data::DataFrame)
    states = reduce(hcat, d[:, 3])
    cm = vcat(
        reduce(hcat, d[:, 4]),
        reduce(hcat, d[:, 5])
    )
    r = reduce(hcat, d[:, 6])
    return states, cm, r
end

# TODO: aggiungere plot ensemble
