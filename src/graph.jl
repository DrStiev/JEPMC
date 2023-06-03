module graph
using Agents, Random, DataFrames
using DrWatson: @dict
using StatsBase: sample, Weights
using Statistics: mean
using Distributions
using CSV, Dates
using UUIDs

@agent Person GraphAgent begin
    status::Symbol # :S, :E, :I, :R
    happiness::Float64 # [-1, 1]
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
    seed = 1234,
)
    rng = Xoshiro(seed)
    C = length(number_point_of_interest)
    # normalizzo il migration rate
    migration_rate_sum = sum(migration_rate, dims = 2)
    for c = 1:C
        migration_rate[c, :] ./= migration_rate_sum[c]
    end
    # scelgo il punto di interesse che avrà il paziente zero
    Is = [zeros(Int, length(number_point_of_interest))...]
    Is[rand(rng, 1:length(Is))] = 1

    # creo il modello 
    model = StandardABM(
        Person,
        GraphSpace(Agents.Graphs.complete_graph(C));
        properties = @dict(
            number_point_of_interest,
            migration_rate,
            new_migration_rate = migration_rate,
            step_count = 0,
            R₀,
            ξ,
            Is,
            C,
            γ,
            σ,
            ω,
            δ,
            η = [zeros(Float64, length(number_point_of_interest))...],
            Rᵢ,
            herd_immunity = (R₀ - 1) / R₀,
        ),
        rng,
    )

    # aggiungo la mia popolazione al modello
    for city = 1:C, _ = 1:number_point_of_interest[city]
        happy = randn(rng)
        happy = happy > 1.0 ? 1.0 : happy < -1.0 ? -1.0 : happy
        add_agent!(
            city,
            model,
            :S,
            happy, # the general happiness of the system will be 0.0
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
        end
    end
    return model
end

function model_step!(model::StandardABM)
    # get info and then apply η
    NAR = update_η(model)
    # balance η due to the happiness of the node
    # in which it is applied
    balance_η_happiness(model, NAR)
    # reduce migration rates in and out NAR
    update_migration_rates!(model, NAR)
    # reduce R₀ due to η
    update!(model)
    # vaccino
    # vaccine!(model)
    # possibilita' di variante
    voc!(model) # variant of concern
    model.step_count += 1
end

slope(x::Number, β = 2) = 1 / (1 + (x / (1 - x))^(-β)) # simil sigmoide.

function balance_η_happiness(model::StandardABM, nar::Vector{Int})
    for n in nar
        avgH = mean([
            h.happiness for h in filter(x -> x.pos == n, [a for a in allagents(model)])
        ])
        # this formula is a very strong assumption
        model.η[n] =
            avgH + model.η[n] < avgH / 2 ? model.η[n] * slope(ratio / abs(avgH), 2) :
            model.η[n]
    end
end

function update_η(model::StandardABM, min_infected = 1)
    function get_node_status(model::StandardABM, pos::Int)
        agents = filter(x -> x.pos == pos, [a for a in allagents(model)])
        infects = filter(x -> x.status == :I, agents)
        return length(infects) / length(agents)
    end

    function calculate_node_risk(model::StandardABM)
        node_at_risk = []
        for c = 1:model.C
            node_status = get_node_status(model, c)
            threshold = min_infected / model.number_point_of_interest[c]
            if node_status ≥ threshold
                push!(node_at_risk, c)
            end
        end
        return node_at_risk
    end

    node_at_risk = calculate_node_risk(model)

    for n in node_at_risk
        s = slope(graph_status[n], 2)
        if model.η[n] == 0.0
            model.η[n] = s
        else
            model.η[n] = model.η[n] > s ? model.η[n] * (1 - s) : s
        end
    end
    return node_at_risk
end

function update_migration_rates!(model::StandardABM, nodes::Vector{Int})
    model.new_migration_rate = model.migration_rate
    for c = 1:model.C
        if in.(c, Ref(nodes))
            model.new_migration_rate[c, :] -= model.migration_rate[c, :] * model.η[c]
            model.new_migration_rate[:, c] -= model.migration_rate[:, c] * model.η[c]
            model.new_migration_rate[c, c] += model.migration_rate[c, c] * model.η[c]
        end
    end
    model.new_migration_rate[model.new_migration_rate.<0.0] .= 0.0
    model.new_migration_rate[model.new_migration_rate.>1.0] .= 1.0
end

function update!(model::StandardABM)
    if model.R₀ > model.Rᵢ
        model.R₀ -= mean(model.η) * (model.R₀ - model.Rᵢ)
    end
end

function vaccine!(model::StandardABM)
    if model.ξ == 0 && rand(model.rng) < 1 / 365
        v = model.herd_immunity / 0.99 # efficacia vaccino
        # voglio arrivare ad avere una herd immunity 
        # entro model.ω tempo
        model.ξ = v / model.ω
    end
end

# very very simple function
function voc!(model::StandardABM)
    # https://www.nature.com/articles/s41579-023-00878-2
    # https://onlinelibrary.wiley.com/doi/10.1002/jmv.27331
    # https://virologyj.biomedcentral.com/articles/10.1186/s12985-022-01951-7
    # nuova variante ogni tot tempo? 
    if rand(model.rng) ≤ 8 * 10E-4 # condizione di attivazione
        # https://it.wikipedia.org/wiki/Numero_di_riproduzione_di_base#Variabilit%C3%A0_e_incertezze_del_R0
        variant = uuid1(model.rng)
        model.R₀ = abs(rand(model.rng, Uniform(3.3, 5.7)))
        model.γ = round(Int, abs(rand(model.rng, Normal(model.γ))))
        model.σ = round(Int, abs(rand(model.rng, Normal(model.σ))))
        model.ω = round(Int, abs(rand(model.rng, Normal(model.ω, model.ω / 10))))
        model.δ = abs(rand(model.rng, Normal(model.δ, model.δ / 10)))
        model.herd_immunity = (model.R₀ - 1) / model.R₀
        # new infects
        new_infects = sample(model.rng, [a for a in allagents(model)], 1)
        for i in new_infects
            i.status = :I
            i.variant = variant
        end
    end
end

function agent_step!(agent, model::StandardABM)
    happiness!(model, agent)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

function happiness!(model::StandardABM, agent)
    agent.happiness += rand(
        model.rng,
        Normal(slope(agent.happiness - model.η[agent.pos], 2), model.η[agent.pos]),
    )
    # mantengo la happiness tra [-1, 1]
    agent.happiness =
        agent.happiness > 1.0 ? 1.0 : agent.happiness < -1.0 ? -1.0 : agent.happiness
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
            (contact.status == :R && !(agent.variant in contact.infected_by))
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
        # fine periodo di latenza
    elseif agent.status == :E && (rand(model.rng) < 1 / model.σ)
        agent.status = :I
        # perdita immunita'
    elseif agent.status == :R && (rand(model.rng) < 1 / model.ω)
        agent.status = :S
    end
end

function recover_or_die!(agent, model::StandardABM)
    # fine malattia
    if agent.status == :I && (rand(model.rng) < 1 / model.γ)
        # probabilità di morte
        if rand(model.rng) < model.δ
            remove_agent!(agent, model)
            return
        end
        # probabilità di guarigione
        agent.status = :R
        push!(agent.infected_by, agent.variant)
        agent.variant = UUID("00000000-0000-0000-0000-000000000000")
    end
end

function collect(
    model::StandardABM;
    astep = agent_step!,
    mstep = model_step!,
    n = 100,
    showprogress = false,
)

    function get_observable_data()
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        happiness(x) = mean(x)

        R₀(model) = model.R₀
        dead(model) = sum(model.number_point_of_interest) - nagents(model)
        active_countermeasures(model) = mean(model.η)

        adata = [
            (:status, susceptible),
            (:status, exposed),
            (:status, infected),
            (:status, recovered),
            (:happiness, happiness),
        ]
        mdata = [dead, R₀, active_countermeasures]
        return adata, mdata
    end

    adata, mdata = get_observable_data()
    ad, md = run!(
        model,
        astep,
        mstep,
        n;
        adata = adata,
        mdata = mdata,
        showprogress = showprogress,
    )

    return hcat(select(ad, Not([:step])), select(md, Not([:step])))
end

function save_dataframe(data::DataFrame, path::String, title = "StandardABM")
    isdir(path) == false && mkpath(path)
    CSV.write(path * title * "_" * string(today()) * ".csv", data)
end

function load_dataset(path::String)
    return DataFrame(CSV.File(path, delim = ",", header = 1))
end
end
