module graph
using Agents, Random, DataFrames
using DrWatson: @dict
using StatsBase: sample, Weights
using InteractiveDynamics
using Statistics: mean
using Distributions
using ProgressMeter
using CSV, Dates

@agent Person GraphAgent begin
    status::Symbol # :S, :E, :I, :R
    happiness::Float64 # [-1, 1]
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
    η::Float64,  # countermeasures speed and effectiveness (0-1)
    seed=42
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
    Is[rand(rng, 1:length(Is))]

    # creo il modello 
    model = StandardABM(
        Person,
        GraphSpace(Agents.Graphs.complete_graph(C));
        properties=@dict(
            number_point_of_interest,
            migration_rate,
            step_count = 0,
            R₀,
            ξ,
            Is,
            C,
            γ,
            σ,
            ω,
            δ,
            η,
            Rᵢ,
        ),
        rng
    )

    # aggiungo la mia popolazione al modello
    for city = 1:C, _ = 1:number_point_of_interest[city]
        add_agent!(city, model, :S, 0.0) # Suscettibile
    end
    # aggiungo il paziente zero
    for city = 1:C
        inds = ids_in_position(city, model)
        for n = 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infetto
        end
    end
    return model
end

function model_step!(model::StandardABM)
    model.step_count += 1
    # get info and then apply η
    # nar = node_at_risk(model, 1E-4)
    # reduce R₀ due to η
    update!(model)
    # possibilita' di variante
    variant!(model)
end

function reduce_migration_rate!(pos::Core.AbstractArray)
    for p in pos
        model.migration_rate[p, :] = 0.0
        model.migration[:, p] = 0.0
        model.migration_rate[p, p] = 1.0
    end
end

function node_at_risk(model::StandardABM; threshold=NaN)
    function get_node_status(model::StandardABM, pos::Int)
        agents = filter(x -> x.pos == pos, [a for a in allagents(model)])
        infects = filter(x -> x.status == :I, agents)
        return length(infects) / length(agents)
    end
    threshold = isnan(threshold) ? 1 / 10^trunc(Int, sum(model.number_point_of_interest)) : 1 / 10^threshold
    graph_status = [get_node_status(model, pos) for pos = 1:model.C]
    return findall(x -> x > threshold, graph_status)
end

function reduce_migration_rates!(model::StandardABM, nodes::Vector{Int})
    for c = 1:model.C
        if in.(c, Ref(nodes))
            model.migration_rate[c, :] = 0.0
            model.migration_rate[c, c] = 1.0
        end
    end
end

function update!(model::StandardABM)
    if model.R₀ > model.Rᵢ
        model.R₀ -= model.η * (model.R₀ - model.Rᵢ)
    end
end

# very very simple function
function variant!(model::StandardABM)
    # https://www.nature.com/articles/s41579-023-00878-2
    # https://onlinelibrary.wiley.com/doi/10.1002/jmv.27331
    # https://virologyj.biomedcentral.com/articles/10.1186/s12985-022-01951-7
    # nuova variante ogni tot tempo? 
    if rand(model.rng) ≤ 8 * 10E-4 # condizione di attivazione
        # https://it.wikipedia.org/wiki/Numero_di_riproduzione_di_base#Variabilit%C3%A0_e_incertezze_del_R0
        model.R₀ = abs(rand(Normal(3.54)))
        model.γ = round(Int, abs(rand(Normal(model.γ))))
        model.σ = round(Int, abs(rand(Normal(model.σ))))
        model.ω = round(Int, abs(rand(Normal(model.ω, model.ω / 10))))
        model.δ = abs(rand(Normal(model.δ, model.δ / 10)))
        # new infects
        new_infects = sample(
            model.rng,
            [a for a in allagents(model)],
            round(Int, length(allagents(model)) * abs(rand(Normal(1E-4, 1E-5)))),
        )
        for i in new_infects
            i.status = :I
        end
    end
end

function agent_step!(agent, model::StandardABM)
    happiness!(agent, -model.η / 10, model.η / 20)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

function happiness!(agent, val::Float64, std::Float64)
    agent.happiness += rand(Normal(val, std))
    # mantengo la happiness tra [-1, 1]
    agent.happiness =
        agent.happiness > 1.0 ? 1.0 : agent.happiness < -1.0 ? -1.0 : agent.happiness
end

function migrate!(agent, model::StandardABM)
    pid = agent.pos
    m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
        happiness!(agent, 0.1, 0.01)
    end
end
# https://github.com/epirecipes/sir-julia/blob/master/markdown/abm/abm.md
function transmit!(agent, model::StandardABM)
    agent.status != :I && return
    ncontacts = rand(model.rng, Poisson(model.R₀))
    for i = 1:ncontacts
        contact = model[rand(model.rng, ids_in_position(agent, model))]
        if contact.status == :S && (rand(model.rng) < model.R₀ / model.γ)
            contact.status = :E
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
    end
end

function collect(model::StandardABM; astep=agent_step!, mstep=model_step!, n=100)

    function get_observable_data()
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        happiness(x) = mean(x)

        R₀(model) = model.R₀
        dead(model) = sum(model.number_point_of_interest) - nagents(model)
        active_countermeasures(model) = model.η

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
    ad, md = run!(model, astep, mstep, n; adata=adata, mdata=mdata, showprogress=true)

    return hcat(select(ad, Not([:step])), select(md, Not([:step])))
end

function save_dataframe(data::DataFrame, path::String, title="StandardABM")
    isdir(path) == false && mkpath(path)
    CSV.write(path * title * "_" * string(today()) * ".csv", data)
end

function load_dataset(path::String)
    return DataFrame(CSV.File(path, delim=",", header=1))
end
end
