# imported libraries
using Agents, Graphs, Random, Distributions, DataFrames
using SparseArrays: findnz
using StatsBase: sample, Weights
using DrWatson: @dict

import OrdinaryDiffEq, DiffEqCallbacks

# included files
include("ABMUtils.jl")
include("Controller.jl")

@agent Node ContinuousAgent{2} begin
    population::Int64
    status::Vector{Float64} # S, E, I, R, D
    param::Vector{Float64} # R₀, γ, σ, ω, δ, η, ξ
    happiness::Float64 # ∈ [0, 1)
end

"""
    function to initialize the model. The model is created as a graph
    init()

    # Arguments
    - numNodes::Int = 50 -> number of nodes to create in the graph
    - edgesCoverage::Symbol = :high -> level of edge coverage across the graph
    - initialNodeInfected::Int = 1 -> initial number of infected node
    - param::Vector = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01] -> vector of parameters
    - avgPopulation::Int = 10_000 -> average number of individuals of each node. This value is used as input of an exponential distribution
    - maxTravelingRate = 0.001 -> maximum value of individuals that can travel between nodes
    - control::Bool = false -> flag indicating the use of non-pharmaceutical control measures
    - vaccine::Bool = false -> flag indicating simulation of pharmaceutical control measures
    dictionary of control options
    - control_options = Dict(:tolerance => 1e-3,
        :dt => 10.0,
        :step => 3.0,
        :maxiters => 100,
        :patience => 3,
        :loss => missing,
        :υ_max => missing)

    # Returns
    - model::ABM -> initialized ABM

"""
function init(;
    numNodes::Int = 50,
    edgesCoverage::Symbol = :high,
    initialNodeInfected::Int = 1,
    param::Vector = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01],
    avgPopulation::Int = 10_000,
    maxTravelingRate = 0.001,
    control::Bool = false,
    vaccine::Bool = false,
    seed::Int = 1234,
    control_options = Dict(:tolerance => 1e-3,
        :dt => 10.0,
        :step => 3.0,
        :maxiters => 100,
        :patience => 10,
        :doplot => false,
        :loss => missing,
        :υ_max => missing))
    rng = Xoshiro(seed)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    graph = connected_graph(numNodes, edgesCoverage; rng = rng)
    migrationMatrix = get_migration_matrix(graph, population, maxTravelingRate)

    properties = @dict(numNodes,
        param,
        graph,
        migrationMatrix,
        step=0,
        control,
        vaccine,
        control_options,
        integrator=nothing)

    model = ABM(Node,
        ContinuousSpace((100, 100); spacing = 4.0, periodic = true);
        properties = properties,
        rng)

    Is = [zeros(Int, numNodes)...]
    for i in 1:initialNodeInfected
        Is[rand(model.rng, 1:numNodes)] = 1
    end

    for node in 1:numNodes
        status = Is[node] == 1 ?
                 [
            (population[node] - 1) / population[node],
            0,
            1 / population[node],
            0,
            0,
        ] :
                 [1.0, 0, 0, 0, 0]
        happiness = rand(model.rng)
        parameters = vcat(param, [0.0, 0.0])
        add_agent!(model, (0, 0), population[node], status, parameters, happiness)
    end

    prob = [OrdinaryDiffEq.ODEProblem(seir!, a.status, (1.0, Inf), a.param)
            for a in allagents(model)]
    integrator = [OrdinaryDiffEq.init(p, OrdinaryDiffEq.Tsit5(); advance_to_tstop = true)
                  for
                  p in prob]
    model.integrator = integrator

    return model
end

"""
    function that updates the model at each step of the simulation
    model_step!(model)
"""
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
    model.vaccine ? vaccine!(model) : nothing
    model.step += 1
end

"""
    function that simulate the research for a pharmaceutical control measure. Additionally simulate the spread of this control measure across the graph when found
    vaccine!(model)
"""
function vaccine!(model::ABM)
    if rand(model.rng) < 1 / 365
        R = mean([agent.param[1] for agent in allagents(model)])
        vaccine = (1 - (1 / R)) / rand(model.rng, Normal(0.83, 0.083))
        vaccine *= mean([agent.param[4] for agent in allagents(model)])
        agent = random_agent(model)
        agent.param[7] = vaccine
    end
    for agent in allagents(model)
        if agent.param[7] > 0.0
            network = model.migrationMatrix[agent.id, :]
            tidxs, tweights = findnz(network)
            id = sample(model.rng, tidxs, Weights(tweights))
            objective = filter(x -> x.id == id, [a for a in allagents(model)])[1]
            objective.param[7] = agent.param[7]
        end
    end
end

"""
    function that updates each agent at every step of the simulation
    agent_step!(agent, model)
"""
function agent_step!(agent, model::ABM)
    migrate!(agent, model)
    happiness!(agent)
    model.control ? control!(agent, model; model.control_options...) : nothing
end

"""
    function that simulate the migration of individual from a node to another one.
    migrate!(agent, model)
"""
function migrate!(agent, model::ABM)
    # get the connections and weights of the matrix
    network = model.migrationMatrix[agent.id, :]
    tidxs, tweights = findnz(network)

    # calculate the amount of individual that travel from each node given the amount of
    # control measure active at each time step. Then update all the relevant information of
    # each agent
    for i in 1:length(tidxs)
        ap = deepcopy(agent.population)
        as = deepcopy(agent.status)

        out = as .* tweights[i] .* (1 - deepcopy(agent.param[6]))
        outp = out .* ap
        new_status = as - out
        new_population = sum(new_status .* ap)
        agent.status = new_status .* ap ./ new_population
        agent.population = round(Int64, new_population)

        objective = filter(x -> x.id == tidxs[i], [a for a in allagents(model)])[1]
        os = deepcopy(objective.status)
        op = deepcopy(objective.population)

        new_status = (os .* op) + outp
        new_population = sum(new_status)
        objective.status = new_status ./ new_population
        objective.population = round(Int64, new_population)
    end
end

"""
    function that updates the happiness of each agent
    happiness!(agent)
"""
function happiness!(agent)
    agent.happiness = -(agent.status[2] + agent.status[3] + agent.status[5]) +
                      (agent.status[1] + (agent.status[4] * (1 - agent.param[6])) -
                       agent.param[6])
    agent.happiness = agent.happiness < 0.0 ? 0.0 :
                      agent.happiness > 1.0 ? 1.0 : agent.happiness
end

"""
    function that simulate the birth of a new Variant of Concearn (VOC)
    voc!(model)
"""
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

"""
    function that enable and update the control of each agent
    control!(agent, model; tolerance = 1e-3, dt = 30.0, step = 7.0, maxiters = 100, loss = missing, υ_max = missing)
"""
function control!(agent,
    model::ABM;
    tolerance = 1e-3,
    dt = 30.0,
    step = 7.0,
    maxiters::Int = 100,
    patience::Int = 3,
    doplot::Bool = false,
    loss = missing,
    υ_max = missing)
    if agent.status[3] ≥ tolerance && model.step % dt == 0
        υ_max = υ_max === missing ?
                Distributions.cdf(Distributions.Beta(2, 5), agent.status[3]) : υ_max
        agent.param[6] = controller(agent.status,
            vcat(agent.param[1:5], agent.param[7]);
            h = agent.happiness,
            timeframe = (0.0, dt),
            step = step,
            maxiters = maxiters,
            patience = patience,
            loss_step = Int(maxiters / 10),
            loss_function = loss,
            υ_max = υ_max,
            doplot = doplot,
            id = agent.id,
            rng = model.rng)
    end
end

"""
    function that collect the results of the simulation
    collect!(model; agent_step = agent_step!, model_step = model_step!, n = 1200, showprogress = false, split_result = true, adata = get_observable_data())
"""
function collect!(model::ABM;
    agent_step = agent_step!,
    model_step = model_step!,
    n::Int = 1200,
    showprogress::Bool = false,
    split_result::Bool = true,
    adata = get_observable_data())
    data, _ = run!(model,
        agent_step,
        model_step,
        n;
        showprogress = showprogress,
        adata = adata)
    if split_result
        return [filter(:id => ==(i), data) for i in unique(data[!, :id])]
    else
        return data
    end
end

"""
    function that collect the results of the simulation when run in the mode `ensemble`
    collect_ensemble!(models; agent_step = agent_step!, model_step = model_step!, n = 1200, showprogress = false, parallel = true, split_result = true, adata = get_observable_data())
"""
function ensemble_collect!(models::Vector;
    agent_step = agent_step!,
    model_step = model_step!,
    n::Int = 1200,
    showprogress::Bool = false,
    parallel::Bool = true,
    adata = get_observable_data(),
    split_result::Bool = true)
    data, _ = ensemblerun!(models,
        agent_step,
        model_step,
        n;
        showprogress = showprogress,
        adata = adata,
        parallel = parallel)
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

"""
    function that collect the results of the simulation when run in the mode `paramscan`
    collect_paramscan!(parameters::Dict = Dict(:maxTravelingRate => Base.collect(0.001:0.003:0.01),
"""
function collect_paramscan!(parameters::Dict = Dict(:maxTravelingRate => Base.collect(0.001:0.003:0.01),
        :edgesCoverage => [:high, :medium, :low],
        :numNodes => Base.collect(4:8:40),
        :initialNodeInfected => Base.collect(1:1:4)),
    init = init;
    adata = get_observable_data(),
    agent_step = agent_step!,
    model_step = model_step!,
    n::Int = 1200,
    showprogress::Bool = false,
    parallel::Bool = true)
    data = paramscan(parameters,
        init;
        adata,
        (agent_step!) = agent_step,
        (model_step!) = model_step,
        n = n,
        showprogress = showprogress,
        parallel = parallel)

    return data
end
