using Agents, Random, Distributions, UUIDs
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

"""
    init([numNodes=Int], [edgesCoverage=Float64], [param=Vector{Float64}], [avgPopulation=Int], [maxTravelingRate=Float64], [tspan=Tuple], [controller=Bool], [seed=Int])

    Initialize the model with the given parameters.
    If none is given, the model will be Initialize with default parameters.

# Examples
```jldoctest
julia> model = init();

```
"""
function init(;
    numNodes::Int=8,
    edgesCoverage::Float64=0.2,
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0],
    avgPopulation::Int=3300,
    maxTravelingRate::Float64=1e-4,  # flusso di persone che si spostano da un nodo all'altro
    controller::Bool=false,
    seed::Int=1234
)
    rng = Xoshiro(seed)
    model = StandardABM(
        Person,
        GraphSpace(graph);
        properties=set_parameters(
            numNodes,
            edgesCoverage,
            param,
            avgPopulation,
            maxTravelingRate,
            controller,
            rng
        ),
        rng
    )

    variant = uuid1(model.rng)
    for city = 1:numNodes, _ = 1:model.population[city]
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
    if controler
        fill(model)
    end
    return model
end


"""
    model_step!(model=StandardABM)

    Function that advance the model by one step. Has different
    behaviour based on the boolean value of the parameter `controller`.
"""
function model_step!(model::StandardABM)
    if model.controller
        fill(model)
        ns = [get_node_status(model, i) for i = 1:model.numNodes]
        controller!(model, ns)
    end
    ABMUtilis.happiness!(model)
    update!(model)
    voc!(model)
    model.step_count += 1
end

"""
    controller!(model=StandardABM, ns=Vector{Float64})

    Function that call all the routines associated to the
    part of controll, prediction and prevention
"""
function controller!(model::StandardABM, ns::Vector{Float64})
    res = predict(model, ns, 30)
    for i in 1:length(res)
        if !isnothing(res[i])
            local_controller!(model, res[i], i, 30; vaccine=model.param[6] > 0.0)
            vaccine!(model, 0.83; time=365)
        end
    end
    global_controller!(model, ns, model.η)
end

"""
    update!(model=StandardABM)

    Function that update the global value of `R₀` based on the
    average value of `η`. `η` represents a vector full of `Float64`
    values ∈ [0, 1] that represent the average value of the active_countermeasures
    applied in a specific node
"""
function update!(model::StandardABM)
    if model.param[1] > 1.0
        model.param[1] -= mean(model.η) * (model.param[1] - 1.0)
    end
end

"""
    agent_step!(agent, model=StandardABM)

    Function that advance the agent by one step
"""
function agent_step!(agent, model::StandardABM)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
end

"""
    migrate!(agent, model=StandardABM)

    Function that migrate each agent from a node to another with a certain
    probability, given by a SparseMatrix `migrationMatrix` created initially
    and associated with the toppology of the graph used to create the model
"""
function migrate!(agent, model::StandardABM)
    pid = agent.pos
    m = sample(model.rng, 1:(model.numNodes), Weights(model.migrationMatrix[pid, :]))
    move_agent!(agent, m, model)
end

"""
    transmit!(agent, model=StandardABM)

    Function that control the contagion. If an agent has status :I
    it means it can infect other agent. The contact ratio between agents is
    governed by a Poisson distribution with λ = `model.param[1]` = R₀.

    This function handle the reinfection from a contact with status = :R
    depending on the type of variant with which is infected. Moreover is
    modeled the possibility to have a resistance based on the similarity between
    variants.
"""
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

"""
    update!(agent, model=StandardABM)

    Function that handle the change in status over time of all the
    agents in the model based on their status and the model.parameter vector
"""
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
