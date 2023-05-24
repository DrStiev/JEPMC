module graph
using Agents, Random, DataFrames
using DrWatson: @dict
using StatsBase: sample, Weights
using InteractiveDynamics
using Statistics: mean
using Distributions
using ProgressMeter

include("controller.jl")

@agent Person GraphAgent begin
    days_infected::Int
    status::Symbol # :S, :E, :I, :R
    happiness::Float64 # [-1, 1]
end

function init(;
    number_point_of_interest,
    migration_rate,
    R₀, # R₀ 
    Rᵢ, # # numero "buono" di riproduzione
    γ,  # periodo infettivita'
    σ,  # periodo esposizione
    ω,  # periodo immunita
    ξ,  # 1 / vaccinazione per milion per day
    δ,  # mortality rate
    η,  # countermeasures speed and effectiveness (0-1)
    seed=1337
)
    rng = Xoshiro(seed)
    C = length(number_point_of_interest)
    # normalizzo il migration rate
    migration_rate_sum = sum(migration_rate, dims=2)
    for c = 1:C
        migration_rate[c, :] ./= migration_rate_sum[c]
    end
    # scelgo il punto di interesse che avrà il paziente zero
    Is = [zeros(Int, length(number_point_of_interest) - 1)..., 1]

    properties = @dict(
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
    )

    # creo il modello 
    model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

    # aggiungo la mia popolazione al modello
    for city = 1:C, _ = 1:number_point_of_interest[city]
        add_agent!(city, model, 0, :S, 0.0) # Suscettibile
    end
    # aggiungo il paziente zero
    for city = 1:C
        inds = ids_in_position(city, model)
        for n = 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infetto
            agent.days_infected = 1
        end
    end
    return model
end

function model_step!(model)
    model.step_count += 1
    update!(model)
    # possibilita' di variante
    variant!(model)
end

function update!(model)
    if model.R₀ > model.Rᵢ
        model.R₀ -= model.η * (model.R₀ - model.Rᵢ)
    end
end

# very very simple function
function variant!(model)
    # https://www.nature.com/articles/s41579-023-00878-2
    # https://onlinelibrary.wiley.com/doi/10.1002/jmv.27331
    # https://virologyj.biomedcentral.com/articles/10.1186/s12985-022-01951-7
    # nuova variante ogni tot tempo? 
    if rand(model.rng) ≤ 8 * 10E-4 # condizione di attivazione
        # https://it.wikipedia.org/wiki/Numero_di_riproduzione_di_base#Variabilit%C3%A0_e_incertezze_del_R0
        newR₀ = rand(Uniform(3.3, 5.7))
        model.R₀ = abs(rand(Normal(newR₀, newR₀ / 10)))
        model.γ = round(Int, abs(rand(Normal(model.γ, model.γ / 10))))
        model.σ = round(Int, abs(rand(Normal(model.σ, model.σ / 10))))
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
            i.days_infected = 1
        end
    end
end

function agent_step!(agent, model)
    happiness!(agent, -model.η / 10, model.η / 20)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

function happiness!(agent, val, std)
    agent.happiness += rand(Normal(val, std))
    # mantengo la happiness tra [-1, 1]
    agent.happiness =
        agent.happiness > 1.0 ? 1.0 : agent.happiness < -1.0 ? -1.0 : agent.happiness
end

function migrate!(agent, model)
    pid = agent.pos
    m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
        happiness!(agent, 0.1, 0.01)
    end
end

function transmit!(agent, model)
    agent.status != :I && return
    n = model.R₀ / model.γ * abs(randn(model.rng))
    n ≤ 0 && return
    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S
            contact.status = :E
            n -= 1
            n ≤ 0 && return
        end
    end
end

function update!(agent, model)
    # possibilita di vaccinazione
    if agent.status == :S
        if rand(model.rng) < model.ξ
            agent.status = :R
        end
        # fine periodo di latenza
    elseif agent.status == :E
        if agent.days_infected > model.σ
            agent.status = :I
        end
        agent.days_infected += 1
        # avanzamento malattia
    elseif agent.status == :I
        agent.days_infected += 1
        # perdita immunita'
    elseif agent.status == :R
        if rand(model.rng) < 1 / model.ω
            agent.status = :S
        end
    end
end

function recover_or_die!(agent, model)
    # fine malattia
    if agent.days_infected > model.γ
        # probabilità di morte
        if rand(model.rng) < model.δ
            remove_agent!(agent, model)
            return
        end
        # probabilità di guarigione
        agent.status = :R
        agent.days_infected = 0
    end
end

function collect(model, astep=agent_step!, mstep=model_step!; n=100, controller_step=7)
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
    df_agent = init_agent_dataframe(model, adata)
    df_model = init_model_dataframe(model, mdata)

    p = if typeof(n) <: Int
        ProgressMeter.Progress(n; enabled=true, desc="run! progress: ")
    else
        ProgressMeter.ProgressUnknown(desc="run! steps done: ", enabled=true)
    end

    s = 0
    while Agents.until(s, n, model)
        if should_we_collect(s, model, true)
            collect_agent_data!(df_agent, model, adata, s)
        end
        if should_we_collect(s, model, true)
            collect_model_data!(df_model, model, mdata, s)
        end
        step!(model, agent_step!, model_step!, 1)
        if mod(s, controller_step) == 0 && s ≠ 0
            controller.countermeasures!(model, df_agent[s-controller_step+1:s, :])
        end
        s += 1
        ProgressMeter.next!(p)
    end
    return hcat(select(df_agent, Not([:step])), select(df_model, Not([:step])))
end
end
