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
        end
    end
    return model
end

function model_step!(model::StandardABM)
    happiness!(model)
    # reduce R₀ due to η
    update!(model)
    # vaccino
    vaccine!(model)
    # possibilita' di variante
    voc!(model) # variant of concern
    model.step_count += 1
end

# need to use a better happiness estimation
function happiness!(model::StandardABM)
    for n = 1:model.C
        # agents = filter(x -> x.pos == n, [a for a in allagents(model)])
        # dead = length(agents) / model.number_point_of_interest[n]
        # infects = filter(x -> x.status == :I, agents)
        # infects = length(infects) / length(agents)
        # very bad estimator for happiness
        model.happiness[n] =
            tanh(model.happiness[n] - model.η[n]) #- tanh(dead + infects) / length(agents))
    end
end

function update!(model::StandardABM)
    if model.R₀ > model.Rᵢ
        model.R₀ -= mean(model.η) * (model.R₀ - model.Rᵢ)
    end
end

function vaccine!(model::StandardABM)
    if model.ξ == 0 && rand(model.rng) < 1 / 365
        # heard immunity over vaccine effectiveness
        v = ((model.R₀ - 1) / model.R₀) / 0.99
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
    recover_or_die!(agent, model)
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
        # gestire vaccinazione e VOC
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

function call_controller(
    model::StandardABM,
    adata::Core.AbstractArray,
    mdata::Core.AbstractArray;
    astep=agent_step!,
    mstep=model_step!,
    n=100,
    showprogress=false,
    tshift=0,
    maxiters=1000,
    initial_training_data=n
)
    training_data = initial_training_data
    i = 0
    res = DataFrame()
    while true
        if i ≥ n
            break
            return res
        end
        # run the model to have a solid base of training data
        ad, md = run!(
            model, # model
            astep, # agent step function
            mstep, # model step function
            training_data - i; # number of steps
            adata=adata, # observable agent data
            mdata=mdata, # model observable data
            showprogress=showprogress # show progress
        )
        res =
            vcat(res[1:end-1, :], hcat(select(ad, Not([:step])), select(md, Not([:step]))))
        i = training_data
        # longterm_est, (predX, guessY), ts = controller.predict(
        (predX, guessY) = controller.predict(
            select(
                res,
                [
                    :susceptible_status,
                    :exposed_status,
                    :infected_status,
                    :recovered_status,
                    :dead,
                ],
            ),
            tshift;
            maxiters
        )
        controller.countermeasures!(model, predX, tshift)
        training_data += tshift
    end
end

function collect(
    model::StandardABM;
    astep=agent_step!,
    mstep=model_step!,
    n=100,
    showprogress=false,
    tshift=0,
    maxiters=1000,
    initial_training_data=n
)

    adata, mdata = get_observable_data()
    if tshift > 0
        return call_controller(
            model,
            adata,
            mdata;
            astep=astep,
            mstep=mstep,
            n=n,
            showprogress=showprogress,
            tshift=tshift,
            maxiters=maxiters,
            initial_training_data=initial_training_data
        )
    else
        ad, md = run!(
            model, # model
            astep, # agent step function
            mstep, # model step function
            n; # number of steps
            adata=adata, # observable agent data
            mdata=mdata, # model observable data
            showprogress=showprogress # show progress
        )
        return hcat(select(ad, Not([:step])), select(md, Not([:step])))
    end

end

function save_dataframe(data::DataFrame, path::String, title="StandardABM")
    isdir(path) == false && mkpath(path)
    CSV.write(path * title * "_" * string(today()) * ".csv", data)
end

function load_dataset(path::String)
    return DataFrame(CSV.File(path, delim=",", header=1))
end
end
