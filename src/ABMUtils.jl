using Graphs, Random, Agents, DataFrames, Dates, CSV, Plots
using Distributions, GraphPlot, Colors, GraphRecipes, StatsPlots
using LinearAlgebra: diagind
using Statistics: mean

adapt_R₀!(x) = return 1.1730158534328545 + 0.21570538523224972 * x

function get_migration_matrix(
    g::SimpleGraph,
    population::Vector{Int},
    maxTravelingRate::Float64,
)
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
            return trunc(
                Int,
                rand(rng, ceil(Int, (avg + low) / 2):floor(Int, (avg + max) / 2)) - low,
            )
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
    for v = 2:n
        add_edge!(g, v, rand(rng, 1:v-1))
    end
    add_random_edges!(g, edge_to_add(n, coverage, rng); rng=rng)

    return g
end

function seir!(du, u, p, t)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ, η, ξ = p
    μ = δ / 1111
    du[1] = μ * sum(u) - R₀ * γ * (1 - η) * S * I + ω * R - ξ * S - μ * S # dS
    du[2] = R₀ * γ * (1 - η) * S * I - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I - μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R + ξ * S - μ * R # dR
    du[5] = δ * γ * I # dD
end

function plot_system_graph(model::ABM)
    max = maximum([agent.population for agent in allagents(model)])
    status = [a.status for a in allagents(model)]
    nodefillc = [
        RGBA(
            1.0 * (status[i][2] + status[i][3] + status[i][5]),
            1.0 * status[i][1],
            1.0 * status[i][4],
            1.0,
        ) for i = 1:length(status)
    ]
    nodelabel = [agent.id for agent in allagents(model)]
    perm = sortperm(nodelabel)
    nodesize = [agent.population for agent in allagents(model)] ./ max
    return graphplot(
        model.graph,
        markersize=0.2,
        node_weights=nodesize,
        names=sort(nodelabel),
        nodeshape=:circle,
        markercolor=nodefillc,
    )
end

function get_observable_data()
    status(x) = x.status
    happiness(x) = x.happiness
    vaccine(x) = x.param[7]
    η(x) = x.param[6]
    R₀(x) = x.param[1]
    return [status, happiness, η, vaccine, R₀]
end

function plot_model(
    data::Vector{DataFrame};
    cumulative::Bool=true,
    ensemble::Bool=false
)
    # TODO: plot ensemble data
    if ensemble
        # get_cumulative_ensemble_data!(data)
    end

    if cumulative
        get_cumulative_plot(data, length(data), length(data[1][!, 1]))
    end
end

function split_dataset(data::DataFrame)
    states = reduce(hcat, data[:, 3])
    cm = vcat(reduce(hcat, data[:, 4]), reduce(hcat, data[:, 5]))
    r = reduce(hcat, data[:, 6])
    return states, cm, r
end

function get_cumulative_plot(data::Vector{DataFrame}, nodes::Int, n::Int)
    l = @layout [
        RecipesBase.grid(1, 1)
        RecipesBase.grid(1, 2)
    ]

    states = 5
    y = fill(NaN, n, nodes, states)
    for i = 1:states
        res = []
        for d in data
            push!(res, reduce(hcat, d[:, 3])'[:, i])
        end
        res = reduce(hcat, res)
        y[:, :, i] = res
    end
    p1 = errorline(1:n, y[:, :, 1], errorstyle=:plume, label="S")
    errorline!(1:n, y[:, :, 2], errorstyle=:plume, label="E")
    errorline!(1:n, y[:, :, 3], errorstyle=:plume, label="I")
    errorline!(1:n, y[:, :, 4], errorstyle=:plume, label="R")
    errorline!(1:n, y[:, :, 5], errorstyle=:plume, label="D")

    states = 2
    y = fill(NaN, n, nodes, states)
    for i = 1:states
        res = []
        for d in data
            push!(res, reduce(hcat, d[:, 3+i]))
        end
        res = reduce(hcat, res)
        y[:, :, i] = res
    end
    p2 = errorline(1:n, y[:, :, 1], errorstyle=:plume, label="happiness")
    errorline!(1:n, y[:, :, 2], errorstyle=:plume, label="countermeasures")
    vax = []
    for d in data
        push!(vax, findfirst(d[:, 6] .!= 0.0))
    end
    vax = filter(x -> !isnothing(x), vax)
    if !isempty(vax)
        v = minimum(vax)
        plot!(p2, [v - 0.01, v + 0.01], [0.0, 1.0], lw=2, color=:green, label=nothing)
        annotate!([(
            v,
            1.0,
            text("Vaccine \nFound", 6, :center, :top, :black, "Helvetica"),
        )])
    end

    states = 1
    y = fill(NaN, n, nodes, states)
    for i = 1:states
        res = []
        for d in data
            push!(res, reduce(hcat, d[:, end]))
        end
        res = reduce(hcat, res)
        y[:, :, i] = res
    end
    p3 = errorline(1:n, y[:, :, 1], errorstyle=:plume, label="R₀")
    plt = plot(
        plot(p1, title="ABM Dynamics", titlefontsize=10),
        plot(p2, title="Agents response to η", titlefontsize=10),
        plot(p3, title="Variant of Concern", titlefontsize=10),
        layout=l,
    )
    return plt
end
