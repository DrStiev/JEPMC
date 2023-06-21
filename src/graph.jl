module graph
using Agents, Random, DataFrames
using DrWatson: @dict
using StatsBase: sample, Weights
using Statistics: mean
using Distributions
using CSV, Dates
using UUIDs

include("controller.jl")

@agent Person GraphAgent begin
    status::Symbol # :S, :E, :I, :R
    variant::UUID
    infected_by::Vector{UUID}
end

function init(;
    number_point_of_interest::Vector{Int},
    migration_rate::Array,
    R₀::Float64, # R₀
    Rᵢ::Float64, # # numero "buono" di riproduzione
    γ::Int,  # periodo infettivita'
    σ::Int,  # periodo esposizione
    ω::Int,  # periodo immunita
    ξ::Float64,  # 1 / vaccinazione per milion per day
    δ::Float64,  # mortality rate
    seed=1234
)
    rng = Xoshiro(seed)
    C = length(number_point_of_interest)
    # normalizzo il migration rate
    migration_rate_sum = sum(migration_rate, dims=2)
    for c = 1:C
        migration_rate[c, :] ./= migration_rate_sum[c]
    end
    # scelgo il punto di interesse che avrà il paziente zero
    Is = [zeros(Int, length(number_point_of_interest))...]
    Is[rand(rng, 1:length(Is))] = 1
    happiness = [randn(rng) for i = 1:C]
    happiness[happiness.<-1.0] .= -1.0
    happiness[happiness.>1.0] .= 1.0

    # creo il modello
    model = StandardABM(
        Person, #agent
        GraphSpace(Agents.Graphs.complete_graph(C)); # space
        properties=@dict(
            number_point_of_interest, # vector
            migration_rate, # matrix
            new_migration_rate = migration_rate, # matrix
            step_count = 0, # counter
            R₀, # float
            R₀ᵢ = R₀,
            ξ, # float ∈ [0,1]
            Is, # vector
            C, # integer
            γ, # integer
            σ, # integer
            ω, # integer
            δ, # float ∈ [0,1]
            η = [zeros(Float64, length(number_point_of_interest))...],
            Rᵢ, # float
            happiness, # vector
            all_variants = [],
            vaccine_coverage = [],
        ),
        rng
    )

    # aggiungo la mia popolazione al modello
    for city = 1:C, _ = 1:number_point_of_interest[city]
        add_agent!(
            city,
            model,
            :S,
            UUID("00000000-0000-0000-0000-000000000000"),
            [UUID("00000000-0000-0000-0000-000000000000")],
        ) # Suscettibile
    end
    # aggiungo il paziente zero
    for city = 1:C
        inds = ids_in_position(city, model)
        for n = 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infetto
            agent.variant = uuid1(rng) # nome variante
            push!(model.all_variants, agent.variant)
        end
    end
    return model
end

function model_step!(model::StandardABM)
    # senza intervento esterno si crea un equilibrio
    # dinamico che apparentemente e' errato ma non lo e'
    happiness!(model)
    # reduce R₀ due to η
    update!(model)
    # new variant of concern
    voc!(model)
    model.step_count += 1
end

# need to use a better happiness estimation
function happiness!(model::StandardABM)
    for n = 1:model.C
        agents = filter(x -> x.pos == n, [a for a in allagents(model)])
        dead = length(agents) / model.number_point_of_interest[n]
        infects = filter(x -> x.status == :I, agents)
        infects = length(infects) / length(agents)
        recovered = filter(x -> x.status == :R, agents)
        recovered = length(recovered) / length(agents)
        # very rough estimator for happiness
        model.happiness[n] = tanh(model.happiness[n] - model.η[n]) +
                             tanh(recovered - (dead + infects)) / 10
    end
end

function update!(model::StandardABM)
    if model.R₀ > model.Rᵢ
        model.R₀ -= mean(model.η) * (model.R₀ - model.Rᵢ)
    end
end

# very very simple function
function voc!(model::StandardABM)
    if rand(model.rng) ≤ 8 * 10E-4 # condizione di attivazione
        variant = uuid1(model.rng)
        model.R₀ = rand(model.rng, Uniform(3.3, 5.7))
        model.γ = round(Int, rand(model.rng, Normal(model.γ, model.γ / 5)))
        model.σ = round(Int, rand(model.rng, Normal(model.σ)))
        model.ω = round(Int, rand(model.rng, Normal(model.ω, model.ω / 10)))
        model.δ = rand(model.rng, Normal(model.δ, model.δ / 10))
        model.R₀ᵢ = model.R₀
        push!(model.all_variants, variant)
        # new infect
        new_infect = random_agent(model)
        new_infect.status = :I
        new_infect.variant = variant
    end
end

function agent_step!(agent, model::StandardABM)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
end

function migrate!(agent, model::StandardABM)
    pid = agent.pos
    m = sample(1:(model.C), Weights(model.new_migration_rate[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

# https://github.com/epirecipes/sir-julia/blob/master/markdown/abm/abm.md
function transmit!(agent, model::StandardABM)
    agent.status != :I && return
    ncontacts = rand(model.rng, Poisson(model.R₀))
    for i = 1:ncontacts
        contact = model[rand(model.rng, ids_in_position(agent, model))]
        if (
            contact.status == :S ||
            (contact.status == :R && !(agent.variant ∈ contact.infected_by))
        ) && (rand(model.rng) < model.R₀ / model.γ)
            contact.status = :E
            contact.variant = agent.variant
        end
    end
end

function update!(agent, model::StandardABM)
    # possibilita di vaccinazione
    if agent.status == :S && (rand(model.rng) < model.ξ)
        agent.status = :R
        agent.infected_by = unique([agent.infected_by; model.vaccine_coverage])
        # fine periodo di latenza
    elseif agent.status == :E && (rand(model.rng) < 1 / model.σ)
        agent.status = :I
        # fine malattia
    elseif agent.status == :I && (rand(model.rng) < 1 / model.γ)
        # probabilità di morte
        if rand(model.rng) < model.δ
            remove_agent!(agent, model)
            return
        end
        # probabilità di guarigione
        agent.status = :R
        push!(agent.infected_by, agent.variant)
        # perdita immunita'
    elseif agent.status == :R && (rand(model.rng) < 1 / model.ω)
        agent.status = :S
    end
end

function get_observable_data()
    susceptible(x) = count(i == :S for i in x)
    exposed(x) = count(i == :E for i in x)
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)

    R₀(model) = model.R₀
    dead(model) = sum(model.number_point_of_interest) - nagents(model)
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
    n=100,
    showprogress=false
)
    adata, mdata = get_observable_data()

    ad, md = run!(
        model, # model
        astep, # agent step function
        mstep, # model step function
        n; # number of steps
        adata=adata, # observable agent data
        mdata=mdata, # model observable data
        showprogress=showprogress # show progress
    )
    AgentsIO.save_checkpoint("data/abm/checkpoint_" * string(today()) * ".jld2", model)
    # AgentsIO.load_checkpoint("data/abm/checkpoint_"*string(today())*".jld2")
    return hcat(select(ad, Not([:step])), select(md, Not([:step])))
end

function ensemble_collect(
    models;
    astep=agent_step!,
    mstep=model_step!,
    n=100,
    showprogress=false,
    parallel=false
)

    # TODO: capire come plottare questi grafici
    adata, mdata = get_observable_data()

    ad, md = ensemblerun!(
        models, # models
        astep, # agent step function
        mstep, # model step function
        n; # number of steps
        adata=adata, # observable agent data
        mdata=mdata, # model observable data
        showprogress=showprogress, # show progress
        parallel=parallel # allow parallelism
    )

    return hcat(select(ad, Not([:step, :ensemble])), select(md, Not([:step])), makeunique=true)
end

function save_dataframe(data::DataFrame, path::String, title="StandardABM")
    isdir(path) == false && mkpath(path)
    CSV.write(path * title * "_" * string(today()) * ".csv", data)
end

function load_dataset(path::String)
    return DataFrame(CSV.File(path, delim=",", header=1))
end
end
