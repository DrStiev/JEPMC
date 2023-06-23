module graph
using Agents, Random, DataFrames
using DrWatson: @dict
using StatsBase: sample, Weights
using Statistics: mean
using Distributions, Distributed
using CSV, Dates
using UUIDs

@agent Person GraphAgent begin
    status::Symbol
    variant::UUID
    infected_by::Vector{UUID}
    variant_tolerance::Int
end

function init(;
    number_point_of_interest::Vector{Int},
    migration_rate::Array,
    R₀::Float64,
    Rᵢ::Float64,
    γ::Int,
    σ::Int,
    ω::Int,
    ξ::Float64,
    δ::Float64,
    seed=1234
)
    rng = Xoshiro(seed)
    C = length(number_point_of_interest)
    migration_rate_sum = sum(migration_rate, dims=2)
    for c = 1:C
        migration_rate[c, :] ./= migration_rate_sum[c]
    end
    Is = [zeros(Int, length(number_point_of_interest))...]
    Is[rand(rng, 1:length(Is))] = 1
    happiness = [randn(rng) for i = 1:C]
    happiness[happiness.<-1.0] .= -1.0
    happiness[happiness.>1.0] .= 1.0

    model = StandardABM(
        Person,
        GraphSpace(Agents.Graphs.complete_graph(C));
        properties=@dict(
            number_point_of_interest,
            migration_rate,
            new_migration_rate = migration_rate,
            step_count = 0,
            R₀,
            R₀ᵢ = R₀,
            ξ,
            Is,
            C,
            γ,
            σ,
            ω,
            δ,
            η = [zeros(Float64, length(number_point_of_interest))...],
            Rᵢ,
            happiness,
            all_variants = [],
            vaccine_coverage = [],
            variant_tolerance = 0,
            # aggiungo campo DataFrame in cui raccolto tutti
            # i dati di ogni passo della simulazione
            # cosi' posso chiamare il controller agilmente
            # dall'esterno 
        ),
        rng
    )

    variant = uuid1(model.rng)
    for city = 1:C, _ = 1:number_point_of_interest[city]
        add_agent!(
            city,
            model,
            :S,
            variant,
            [variant],
            0,
        )
    end
    for city = 1:C
        inds = ids_in_position(city, model)
        for n = 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I
            agent.variant = uuid1(rng)
            push!(model.all_variants, agent.variant)
        end
    end
    return model
end

function model_step!(model::StandardABM)
    happiness!(model)
    update!(model)
    voc!(model)
    # controller.controller_vaccine!(model, 0.83; time=365)
    model.step_count += 1
end

function happiness!(model::StandardABM)
    for n = 1:model.C
        agents = filter(x -> x.pos == n, [a for a in allagents(model)])
        dead = (length(agents) - model.number_point_of_interest[n]) / model.number_point_of_interest[n]
        infects = filter(x -> x.status == :I, agents)
        infects = length(infects) / length(agents)
        recovered = filter(x -> x.status == :R, agents)
        recovered = length(recovered) / length(agents)
        model.happiness[n] =
            tanh((model.happiness[n] - model.η[n]) + (recovered/3 - (dead + infects)))
        # if model.step_count % model.γ == 0
        #     controller.controller_happiness!(model)
        # end
        model.happiness[n] = model.happiness[n] > 1.0 ? 1.0 :
                             model.happiness[n] < -1.0 ? -1.0 : model.happiness[n]
    end
end

function update!(model::StandardABM)
    if model.R₀ > model.Rᵢ
        model.R₀ -= mean(model.η) * (model.R₀ - model.Rᵢ)
    end
end

function voc!(model::StandardABM)
    if rand(model.rng) ≤ 8 * 10E-4
        variant = uuid1(model.rng)
        model.R₀ = rand(model.rng, Uniform(3.3, 5.7))
        model.γ = round(Int, rand(model.rng, Normal(model.γ, model.γ / 5)))
        model.σ = round(Int, rand(model.rng, Normal(model.σ)))
        model.ω = round(Int, rand(model.rng, Normal(model.ω, model.ω / 10)))
        model.δ = rand(model.rng, Normal(model.δ, model.δ / 10))
        model.R₀ᵢ = model.R₀
        push!(model.all_variants, variant)

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

function coverage(s1::UUID, ss2::Vector{UUID}, maxdiff::Int)
    new_s1 = string(s1)
    new_ss2 = string.(ss2)
    for j = 1:length(new_ss2)
        dist = 0
        for i = 1:8
            @inbounds dist += abs(new_s1[i] - new_ss2[j][i])
        end
        if dist > maxdiff
            return false
        end
    end
    return true
end

function migrate!(agent, model::StandardABM)
    pid = agent.pos
    m = sample(1:(model.C), Weights(model.new_migration_rate[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent, model::StandardABM)
    agent.status != :I && return
    ncontacts = rand(model.rng, Poisson(model.R₀))
    for i = 1:ncontacts
        contact = model[rand(model.rng, ids_in_position(agent, model))]
        if (
            contact.status == :S ||
            (contact.status == :R && !(agent.variant ∈ contact.infected_by) &&
             !coverage(agent.variant, contact.infected_by, contact.variant_tolerance))
        ) && (rand(model.rng) < model.R₀ / model.γ)
            contact.status = :E
            contact.variant = agent.variant
        end
    end
end

function update!(agent, model::StandardABM)
    if agent.status == :S && (rand(model.rng) < model.ξ)
        agent.status = :R
        agent.infected_by = unique([agent.infected_by; model.vaccine_coverage])
        agent.variant_tolerance = model.variant_tolerance
    elseif agent.status == :E && (rand(model.rng) < 1 / model.σ)
        agent.status = :I
    elseif agent.status == :I && (rand(model.rng) < 1 / model.γ)
        if rand(model.rng) < model.δ
            remove_agent!(agent, model)
            return
        end
        agent.status = :R
        push!(agent.infected_by, agent.variant)
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
        model,
        astep,
        mstep,
        n;
        adata=adata,
        mdata=mdata,
        showprogress=showprogress
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
