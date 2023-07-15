using Agents, Graphs, Random, Distributions, DataFrames
using SparseArrays: findnz
using StatsBase: sample, Weights

import OrdinaryDiffEq, DiffEqCallbacks

include("ABMUtils.jl")
include("Controller.jl")

@agent Node ContinuousAgent{2} begin
    population::Int
    status::Vector{Float64} # S, E, I, R, D
    param::Vector{Float64} # R₀, γ, σ, ω, δ, η, ξ
    happiness::Float64
end

function init(;
    numNodes::Int=50,
    edgesCoverage::Symbol=:high,
    initialNodeInfected::Int=1,
    param::Vector{Float64}=[3.54, 1 / 14, 1 / 5, 1 / 280, 0.007],
    avgPopulation::Int=10000,
    maxTravelingRate::Float64=0.1,  # flusso di persone che si spostano
    tspan::Tuple=(1.0, Inf),
    control::Bool=false,
    threshold::Float64=1e-3,
    seed::Int=1234
)

    rng = Xoshiro(seed)
    population = map((x) -> round(Int, x), randexp(rng, numNodes) * avgPopulation)
    graph = connected_graph(numNodes, edgesCoverage; rng=rng)
    migrationMatrix = get_migration_matrix(graph, population, maxTravelingRate)

    model = ABM(
        Node,
        ContinuousSpace((100, 100); spacing=4.0, periodic=true);
        properties=Dict(
            :numNodes => numNodes,
            :param => param,
            :connections => graph,
            :migrationMatrix => migrationMatrix,
            :step => 0,
            :control => control,
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

    prob = [OrdinaryDiffEq.ODEProblem(seir!, a.status, tspan, a.param) for a in allagents(model)]
    integrator = [OrdinaryDiffEq.init(p, OrdinaryDiffEq.Tsit5(); advance_to_tstop=true) for p in prob]
    model.properties[:integrator] = integrator

    return model
end

function model_step!(model::ABM; show::Bool=false)
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
    if show
        display(plot_system_graph(model))
    end
    model.step += 1
end

function agent_step!(agent, model::ABM)
    migrate!(agent, model)
    happiness!(agent)
    model.control ? controller!(agent, model) : nothing
end

function migrate!(agent, model::ABM)
    network = model.migrationMatrix[agent.id, :]
    tidxs, tweights = findnz(network)

    for i in 1:length(tidxs)
        try
            people_traveling_out = map((x) -> round(Int, x), agent.status .* tweights[i] .* agent.population .* (1 - agent.param[6]))
            dead = people_traveling_out[end]
            people_traveling_out[end] = 0.0

            new_population = agent.population - (sum(people_traveling_out) + dead)
            agent.status = (agent.status .* agent.population - people_traveling_out) ./ new_population
            agent.population = new_population
            agent.population = map((x) -> round(Int, x), agent.population)

            objective = filter(x -> x.id == tidxs[i], [a for a in allagents(model)])[1]
            new_population = objective.population + sum(people_traveling_out)
            objective.status = (objective.status .* objective.population + people_traveling_out) ./ new_population
            objective.population = new_population
            objective.population = map((x) -> round(Int, x), objective.population)
        catch ex
            @warn "Something was odd: " exception = (ex, catch_backtrace())
        end
    end
end

# TODO: vedere se mantenere campo happiness nel caso, migliorare stimatore
function happiness!(agent)
    agent.happiness = tanh(agent.happiness - agent.param[6] + (agent.status[4] - (agent.status[5] + agent.status[3])))
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

function controller!(agent, model::ABM)
    # TODO: implementare eliminazione di queste policy?
    if agent.status[3] ≥ 1e-3 && model.step % 30 == 0
        agent.param[6] = controller!(agent.status, agent.param)[1]
    end
end

function collect!(
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

function ensemble_collect!(
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

n = 300
model = init(; numNodes=4, avgPopulation=2000, control=true)
plot_system_graph(model)
data = collect!(model; n=n)
plt = plot_model(data; cumulative=true)
include("Utils.jl")
save_plot(plt, "img/abm_ode/", "SocialNetworkABM", "pdf")

# TODO: param scan su :maxTravelingRate, :edgesCoverage, :numNodes, :avgPopulation
# TODO: cacciare un modo per visualizzare decentemente i grafici soprattutto questi complessi
# sbattere un bel @everywhere nell'include e via. fare sti test file separato
parameters = Dict(
    :maxTravelingRate => Base.collect(0.01:0.01:0.1),
    :edgesCoverage => [:high, :medium, :low],
    :numNodes => Base.collect(1:10:101),
    :avgPopulation => Base.collect(1000:1000:100000),
)

r = paramscan(
    parameters,
    init;
    adata=get_observable_data(),
    (agent_step!)=agent_step!,
    (model_step!)=model_step!,
    n=1200,
    showprogress=true,
    parallel=true
)

plot_model(data; cumulative=true)

models = [init(; seed=abs(i)) for i in rand(Int64, 10)]
data = ensemble_collect!(models; n=n)

for d in data
    display(plot_model(d))
end
