using Agents, Random, Distributions, UUIDs
using DrWatson: @dict
using StatsBase: sample, Weights
using Statistics: mean

include("ABMUtilis.jl")
include("Controller.jl")

@agent Person GraphAgent begin
    status::Symbol
    variant::UUID
    infected_by::Vector{UUID}
    variant_tolerance::Int
end

function init(;
    numNodes::Int=50,
    edgesCoverage::Float64=0.2,
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0],
    avgPopulation::Int=3300,
    maxTravelingRate::Float64=0.1,  # flusso di persone che si spostano
    tspan::Tuple=(1.0, Inf),
    controller::Bool=false,
    seed::Int=1234
)
    rng = Xoshiro(seed)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    graph = generate_nearly_complete_graph(numNodes, floor(Int, (1 - edgesCoverage) * (numNodes * (numNodes - 1) / 2)); seed=seed)
    migrationMatrix = get_migration_matrix(graph, population, numNodes, maxTravelingRate)

    Is = [zeros(Int, numNodes)...]
    Is[rand(rng, 1:length(Is))] = 1

    happiness = randn(numNodes)
    happiness[happiness.>1.0] .= 1.0
    happiness[happiness.<-1.0] .= -1.0

    model = StandardABM(
        Person,
        GraphSpace(graph);
        properties=@dict(
            numNodes,
            migrationMatrix,
            population,
            param,
            η = [zeros(Float64, numNodes)...],
            controller,
            all_variants = [],
            vaccine_coverage = [],
            variant_tolerance = 0,
            happiness,
            outresults = DataFrame(
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
        ),
        rng
    )

    variant = uuid1(model.rng)
    for city = 1:numNodes, _ = 1:population[city]
        add_agent!(city, model, :S, variant, [variant], 0)
    end
    for city = 1:numNodes
        inds = ids_in_position(city, model)
        for n = 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I
            agent.variant = uuid1(rng)
            push!(model.all_variants, agent.variant)
        end
    end
    for i = 1:numNodes
        node = filter(x -> x.pos == i, [a for a in allagents(model)])
        push!(
            model.outresults,
            [
                length(filter(x -> x.status == :S, node)),
                length(filter(x -> x.status == :E, node)),
                length(filter(x -> x.status == :I, node)),
                length(filter(x -> x.status == :R, node)),
                length(node) - sum(population[i]),
                model.param[1],
                model.η[i],
                model.happiness[i],
                i,
            ],
        )
    end
    return model
end

function model_step!(model::StandardABM)
    if model.controller
        collect!(model)
        ns = [controller.get_node_status(model, i) for i = 1:model.numNodes]
        controller!(model, ns)
    end
    happiness!(model)
    update!(model)
    voc!(model)
    model.step_count += 1
end

function controller!(model::StandardABM, ns::Vector{Float64})
    res = controller.predict(model, ns, 30)
    for i in 1:length(res)
        if !isnothing(res[i])
            controller.local_controller!(model, res[i], i, 30; vaccine=model.param[6] > 0.0)
            controller.vaccine!(model, 0.83; time=365)
        end
    end
    controller.global_controller!(model, ns, model.η)
end

function update!(model::StandardABM)
    if model.param[1] > 1.0
        model.param[1] -= mean(model.η) * (model.param[1] - 1.0)
    end
end

function agent_step!(agent, model::StandardABM)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
end

function migrate!(agent, model::StandardABM)
    pid = agent.pos
    m = sample(model.rng, 1:(model.numNodes), Weights(model.migrationMatrix[pid, :]))
    move_agent!(agent, m, model)
end

function transmit!(agent, model::StandardABM)
    agent.status != :I && return
    ncontacts = rand(model.rng, Poisson(model.param[1]))
    for i = 1:ncontacts
        contact = model[rand(model.rng, ids_in_position(agent, model))]
        if (
            contact.status == :S || (
                contact.status == :R &&
                !(agent.variant ∈ contact.infected_by) &&
                !coverage(agent.variant, contact.infected_by, contact.variant_tolerance)
            )
        ) && (rand(model.rng) < model.param[1] / model.param[2])
            contact.status = :E
            contact.variant = agent.variant
        end
    end
end

function update!(agent, model::StandardABM)
    if agent.status == :S && (rand(model.rng) < model.param[6])
        agent.status = :R
        agent.infected_by = unique([agent.infected_by; model.vaccine_coverage])
        agent.variant_tolerance = model.variant_tolerance
    elseif agent.status == :E && (rand(model.rng) < 1 / model.param[3])
        agent.status = :I
    elseif agent.status == :I && (rand(model.rng) < 1 / model.param[2])
        if rand(model.rng) < model.param[5]
            remove_agent!(agent, model)
            return
        end
        agent.status = :R
        push!(agent.infected_by, agent.variant)
    elseif agent.status == :R && (rand(model.rng) < 1 / model.param[4])
        agent.status = :S
    end
end
