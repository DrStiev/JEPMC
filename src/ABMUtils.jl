### -*- Mode: Julia -*-

### ABMUtils.jl
###
### See file LICENSE in top folder for copyright and licensing
### information.

using Graphs, Random, Agents, DataFrames, Dates, CSV, Plots
using Distributions, GraphPlot, Colors, GraphRecipes, StatsPlots
using DifferentialEquations, SciMLSensitivity
using LinearAlgebra: diagind
using Statistics: mean

gr()

# adapt_R₀!(x) = return 1.1730158534328545 + 0.21570538523224972 * x

"""
    function that computes the migration matrix of a graph
    get_migration_matrix(g::SimpleGraph, population::Vector{Int}, maxTravelingRate::Float64)
"""
function get_migration_matrix(g::SimpleGraph,
    population::Vector{Int},
    maxTravelingRate::Float64)
    numNodes = Graphs.nv(g)
    migrationMatrix = (population .+ population') ./ population
    migrationMatrix = (migrationMatrix .* maxTravelingRate) ./ maximum(migrationMatrix)
    migrationMatrix[diagind(migrationMatrix)] .= 1.0
    mmSum = sum(migrationMatrix, dims = 2)
    migrationMatrix ./= mmSum
    return migrationMatrix .* adjacency_matrix(g)
end

"""
    function that creates a connected graph with a specific arc coverage
    connected_graph(n::Int, coverage::Symbol; rng::AbstractRNG)
"""
function connected_graph(n::Int, coverage::Symbol; rng::AbstractRNG)
    function edge_to_add(n::Int, coverage::Symbol, rng::AbstractRNG)
        low = n - 1
        if coverage == :low
            avg = (n * (n - 1) / 2 + (n - 1)) / 2
            return trunc(Int, rand(rng, low:((avg + low) / 2)))
        elseif coverage == :medium
            avg = (n * (n - 1) / 2 + (n - 1)) / 2
            max = n * (n - 1) / 2
            return trunc(Int, rand(rng, ((avg + low) / 2):((avg + max) / 2)))
        elseif coverage == :high
            avg = (n * (n - 1) / 2 + (n - 1)) / 2
            max = n * (n - 1) / 2
            return trunc(Int, rand(rng, ((avg + max) / 2):max))
        end
    end

    function add_random_edges!(graph::SimpleGraph, n::Int; rng::AbstractRNG)
        for i in 1:n
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
        add_edge!(g, v, rand(rng, 1:(v - 1)))
    end
    add_random_edges!(g, edge_to_add(n, coverage, rng); rng = rng)

    return g
end

"""
    function that describe an SEIR(S) model
    seir!(du, u, p, t)
"""
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

"""
    function that plot the visual representation of a graph using GraphRecipes given a ABM
    plot_system_graph(model::ABM)
"""
function plot_system_graph(model::ABM)
    max = maximum([agent.population for agent in allagents(model)])
    status = [a.status for a in allagents(model)]
    nodefillc = [RGBA(status[i][2] + status[i][3], # R
        status[i][1], # G
        status[i][4], # B
        1.0 - status[i][5]) for i in eachindex(status)]
    nodelabel = [agent.id for agent in allagents(model)]
    perm = sortperm(nodelabel)
    nodesize = [agent.population / max for agent in allagents(model)]
    return GraphRecipes.graphplot(model.graph,
        method = :shell, # otherwise it change position everytime it's been plot
        markersize = 0.2,
        node_weights = nodesize,
        names = sort(nodelabel),
        nodeshape = :circle,
        markercolor = nodefillc)
end

"""
    function that returns the observable data of the model
"""
function get_observable_data()
    status(x) = x.status
    happiness(x) = x.happiness
    υ(x) = x.param[7]
    η(x) = x.param[6]
    R₀(x) = x.param[1]
    return [status, happiness, η, υ, R₀]
end

"""
    function that return the plot of all the relevant information of the model given a DataFrame of data.
    There are 3 main information:
        - SEIR model
        - happiness over countermeasures
        - R₀ index
    plot_model(data; errorstyle = :ribbon, title::String = "")
"""
function plot_model(data; errorstyle = :ribbon, title::String = "")
    # If data is a single DataFrame, convert it to an array of DataFrames
    if typeof(data) <: DataFrame
        data = [data]
    end
    get_cumulative_plot(data,
        length(data),
        length(data[1][!, 1]);
        errorstyle = errorstyle,
        title = title)
end

function get_cumulative_plot(data::Vector{DataFrame},
    nodes::Int,
    n::Int;
    errorstyle = :plume,
    title::String = "")
    l = @layout [RecipesBase.grid(1, 1)
        RecipesBase.grid(1, 2)]

    states = 5
    y = fill(NaN, n, nodes, states)
    for i in 1:states
        res = []
        for d in data
            push!(res, reduce(hcat, d[:, 3])'[:, i])
        end
        res = reduce(hcat, res)
        y[:, :, i] = res
    end
    p1 = errorline(1:n, y[:, :, 1], errorstyle = :plume, label = "S")
    errorline!(1:n, y[:, :, 2], errorstyle = :plume, label = "E")
    errorline!(1:n, y[:, :, 3], errorstyle = :plume, label = "I")
    errorline!(1:n, y[:, :, 4], errorstyle = :plume, label = "R")
    errorline!(1:n, y[:, :, 5], errorstyle = :plume, label = "D")

    states = 2
    y = fill(NaN, n, nodes, states)
    for i in 1:states
        res = []
        for d in data
            push!(res, reduce(hcat, d[:, 3 + i]))
        end
        res = reduce(hcat, res)
        y[:, :, i] = res
    end
    p2 = errorline(1:n, y[:, :, 1], errorstyle = errorstyle, label = "happiness")
    errorline!(1:n, y[:, :, 2], errorstyle = errorstyle, label = "countermeasures")
    vax = unique(collect(Iterators.flatten([unique(filter(x -> x .!= 0.0, d[:, 6]))
                                            for d in data])))
    vax = sort(unique(filter(x -> !isnothing(x),
        [findfirst(d[:, 6] .== v) for v in vax for d in data])))
    if !isempty(vax)
        label = "vaccine"
        for v in vax
            plot!(p2,
                [v - 0.01, v + 0.01],
                [0.0, 1.0],
                lw = 3,
                color = :green,
                label = label)
            label = nothing
        end
    end

    states = 1
    y = fill(NaN, n, nodes, states)
    for i in 1:states
        res = []
        for d in data
            push!(res, reduce(hcat, d[:, end]))
        end
        res = reduce(hcat, res)
        y[:, :, i] = res
    end
    p3 = errorline(1:n, y[:, :, 1], errorstyle = errorstyle, label = "R₀")
    plt = plot(plot(p1, title = "ABM Dynamics " * title, titlefontsize = 10),
        plot(p2, title = "Agents response to η", titlefontsize = 10),
        plot(p3, title = "Variant of Concern", titlefontsize = 10),
        layout = l)
    return plt
end

# need generalization
"""
    function that return the sensitivity of the model about certain parameters
    sensitivity_analisys(f, u0, tspan, p; doplot::Bool = true)
"""
function sensitivity_analisys(f, u0, tspan, p; doplot::Bool = true)
    prob = ODEForwardSensitivityProblem(f, u0, tspan, p)
    sol = solve(prob, Tsit5())
    x, dp = extract_local_sensitivities(sol)
    pltout = nothing

    if doplot
        plt = []
        truesol = solve(ODEProblem(f, u0, tspan, p), Tsit5())
        push!(plt,
            plot(truesol, lw = 2, title = "Data", titlefontsize = 10, legend = false))

        for i in 1:length(p)
            push!(plt,
                plot(sol.t,
                    dp[i]',
                    lw = 2,
                    title = "Sensitivity to param $i",
                    titlefontsize = 10,
                    legend = false))
        end

        pltout = plot(plt...)
    end

    return x, dp, pltout
end
### end of file -- ABMUtils.jl
