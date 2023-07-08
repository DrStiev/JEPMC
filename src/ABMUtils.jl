using Graphs, Random, Agents, DataFrames, Dates, CSV
using LinearAlgebra: diagind
using Statistics: mean

include("Utils.jl")

# funzione per adattare R₀ del modello ODE al funzionamento
# bizzarro del modello ad agente. Vedi grafici in img/abm, img/ode, img/abm_ode
adapt_R₀!(x) = return 1.1730158534328545 + 0.21570538523224972 * x

Base.@kwdef mutable struct Parameters
    numNodes::Int
    migrationMatrix::AbstractMatrix
    population::Vector{Int}
    param::Vector{Float64}
    η::Vector{Float64}
    controller::Bool
    all_variants::Vector{UUID}
    vaccine_coverage::Vector{Float64}
    variant_tolerance::Int
    happiness::Vector{Float64}
    outresults::DataFrame
    step::Int
end

function set_parameters(
    graph::SimpleGraph,
    param::Vector{Float64},
    avgPopulation::Int,
    maxTravelingRate::Float64,  # flusso di persone che si spostano
    controller::Bool,
    rng::AbstractRNG,
)

    numNodes = Graphs.nv(graph)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    migrationMatrix = get_migration_matrix(graph, population, maxTravelingRate)

    happiness = randn(numNodes)
    happiness[happiness.>1.0] .= 1.0
    happiness[happiness.<-1.0] .= -1.0

    res = Parameters(
        numNodes,
        migrationMatrix,
        population,
        param,
        [zeros(Float64, numNodes)...],
        controller,
        [],
        [],
        0,
        happiness,
        DataFrame(
            susceptible=Int[],
            exposed=Int[],
            infected=Int[],
            recovered=Int[],
            dead=Int[],
            R0=Float64[],
            active_countermeasures=Float64[],
            happiness=Float64[],
            node=Int[],
        ),
        1
    )

    return res
end

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

function generate_nearly_complete_graph(n::Int, coverage::Symbol; rng::AbstractRNG)
    function edge_to_remove(n::Int, coverage::Symbol, rng::AbstractRNG)
        low = n - 1
        max = n * (n - 1) / 2
        avg = (n * (n - 1) / 2 + (n - 1)) / 2
        if coverage == :low
            return trunc(Int, max - rand(rng, low:floor(Int, (avg + low) / 2)))
        elseif coverage == :medium
            return trunc(Int, max - rand(rng, ceil(Int, (avg + low) / 2):floor(Int, (avg + max) / 2)))
        elseif coverage == :high
            return trunc(Int, max - rand(rng, ceil(Int, (avg + max) / 2):max))
        end
    end

    g = complete_graph(n)

    # Remove N random edges
    e = Graphs.collect(Graphs.edges(g))
    shuffled_edges = e[randperm(rng, length(e))]
    edges_to_remove = shuffled_edges[1:edge_to_remove(n, coverage, rng)]
    for e in edges_to_remove
        Graphs.rem_edge!(g, e)
    end

    return g
end

function coverage(s1::UUID, ss2::Vector{UUID}, maxdiff::Int)
    new_s1 = string(s1)
    new_ss2 = string.(ss2)
    for j = 1:length(new_ss2)
        dist = 0
        for i = 1:8
            dist += abs(new_s1[i] - new_ss2[j][i])
        end
        if dist > maxdiff
            return false
        end
    end
    return true
end

function happiness!(model::StandardABM)
    for n = 1:model.numNodes
        agents = filter(x -> x.pos == n, [a for a in allagents(model)])
        dead =
            (length(agents) - model.population[n]) /
            model.population[n]
        infects = filter(x -> x.status == :I, agents)
        infects = length(infects) / length(agents)
        recovered = filter(x -> x.status == :R, agents)
        recovered = length(recovered) / length(agents)
        model.happiness[n] =
            tanh((model.happiness[n] - model.η[n]) + (recovered / 3 - (dead + infects)))
        model.happiness[n] =
            model.happiness[n] > 1.0 ? 1.0 :
            model.happiness[n] < -1.0 ? -1.0 : model.happiness[n]
    end
end

function fill(model::StandardABM)
    for i = 1:model.numNodes
        node = filter(x -> x.pos == i, [a for a in allagents(model)])
        push!(
            model.outresults,
            [
                length(filter(x -> x.status == :S, node)),
                length(filter(x -> x.status == :E, node)),
                length(filter(x -> x.status == :I, node)),
                length(filter(x -> x.status == :R, node)),
                length(node) - sum(model.population[i]),
                model.param[1],
                model.η[i],
                model.happiness[i],
                i,
            ],
        )
    end
end

function voc!(model::StandardABM)
    if rand(model.rng) ≤ 8E-3
        variant = uuid1(model.rng)
        model.param[1] = rand(model.rng, Uniform(3.3, 5.7))
        model.param[2] = round(Int, rand(model.rng, Normal(model.param[2], model.param[2] / 10)))
        model.param[3] = round(Int, rand(model.rng, Normal(model.param[3], model.param[3] / 10)))
        model.param[4] = round(Int, rand(model.rng, Normal(model.param[4], model.param[4] / 10)))
        model.param[5] = rand(model.rng, Normal(model.param[5], model.param[5] / 10))

        push!(model.all_variants, variant)

        new_infect = random_agent(model)
        new_infect.status = :I
        new_infect.variant = variant
    end
end


function get_observable_data()
    susceptible(x) = count(i == :S for i in x)
    exposed(x) = count(i == :E for i in x)
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)

    R₀(model) = model.param[1]
    dead(model) = sum(model.population) - nagents(model)
    active_countermeasures(model) = mean(model.η)
    happiness(model) = mean(model.happiness)

    adata = [
        (:status, susceptible),
        (:status, exposed),
        (:status, infected),
        (:status, recovered),
    ]
    mdata = [dead, R₀, active_countermeasures, happiness]
    return adata, mdata
end

function collect(
    model::StandardABM;
    astep=agent_step!,
    mstep=model_step!,
    n::Int=100,
    showprogress::Bool=true
)
    adata, mdata = get_observable_data()

    ad, md = run!(
        model,
        astep,
        mstep,
        n;
        adata=adata,
        mdata=mdata,
        showprogress=showprogress
    )
    # AgentsIO.save_checkpoint("data/abm/checkpoint_" * string(today()) * ".jld2", model)
    # AgentsIO.load_checkpoint("data/abm/checkpoint_"*string(today())*".jld2")
    res = hcat(select(ad, Not([:step])), select(md, Not([:step])))
    rename!(res, [:susceptible, :exposed, :infected, :recovered, :dead, :R0, :active_countermeasures, :happiness])
    return res
    # return model.outresults
end

function ensemble_collect(
    models;
    astep=agent_step!,
    mstep=model_step!,
    n::Int=100,
    showprogress::Bool=false,
    parallel::Bool=false,
    split_result::Bool=true
)

    adata, mdata = get_observable_data()

    ad, md = ensemblerun!(
        models,
        astep,
        mstep,
        n;
        adata=adata,
        mdata=mdata,
        showprogress=showprogress,
        parallel=parallel
    )
    res = hcat(
        select(ad, Not([:step, :ensemble])),
        select(md, Not([:step])),
        makeunique=true,
    )
    if split_result
        return [filter(:ensemble => ==(i), res) for i in unique(res[!, :ensemble])]
    else
        return res
    end
end
