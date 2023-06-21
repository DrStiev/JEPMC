module controller

using Agents, DataFrames, Random, Distributions
using Statistics: mean

# https://github.com/epirecipes/sir-julia

include("utils.jl")
# parametri su cui il controllore può agire:
# η → countermeasures (0.0 - 1.0)
# Rᵢ → objective value for R₀
# ξ → vaccination rate

function controller_vaccine!(model::StandardABM, avg_effectiveness::Float64; time=365)
    # poco realistico ma funzionale
    # aggiornamento vaccino oppure nuovo vaccino
    if rand(model.rng) < 1 / time
        # heard immunity over vaccine effectiveness
        v =
            (1 - (1 / model.R₀ᵢ)) /
            rand(model.rng, Normal(avg_effectiveness, avg_effectiveness / 10))
        # voglio arrivare ad avere una herd immunity
        # entro model.ω tempo
        model.ξ = v / model.ω
        model.vaccine_coverage = model.all_variants
    end
    # TODO: usando un uuid1 i valori che cambiano sono sempre e soli i primi 8
    # questo a "codificare" una variante di uno specifico virus. se la variante
    # non è troppo distante da quelle inserite nella copertura vaccinale si è 
    # resistenti anche a quella. ognuna delle 8 posizioni può assumere un 
    # valore hex in [0-f]. posso calcolare una edit distance custom
    # definendo come pesare la differenza tra due valori. Questa differenza
    # viene poi pesata per la tolleranza insita del vaccino.
    # s1 = xxxxxxxx-..., s2 = xxxxxxxx-... sum(abs(x1i-x2i)) 
    # diff in [0-128]. se diff <= abstol allora si è coperti. 
    # abstol è dato dal tempo di sviluppo del vaccino (+ tempo + abstol) 
    # * effectiveness. abstol = tanh(model.step/time*avg_effectiveness)
    # max_diff = round(Int, abstol*128)
    
end

function controller_η!(model::StandardABM, data::Matrix{Int}, step::Int; mininfects=1)

    # funzione si occupa di applicare delle contromisure
    # al modello in base alla sua situazione corrente rispetto
    # a quella di N passi precedenti.

    # control over row 3 and 5 for status :I and :D
    # return the slope of a tanh (- if - and + if +)
    slope(pred) = tanh(pred[3, end] - pred[3, 1]) + tanh(pred[5, end] - pred[5, 1])

    # get the infection rate for each node
    function get_node_status(model::StandardABM, pos::Int)
        agents = filter(x -> x.pos == pos, [a for a in allagents(model)])
        infects = filter(x -> x.status == :I, agents)
        # return length(infects) / length(agents)
        # iff there are a minimum number of infections
        if length(infects) > mininfects
            return length(infects) / length(agents)
        else
            return 0.0
        end
    end

    function apply_lockdown!(model::StandardABM, node::Int, restriction::Float64)
        # update the migration matrix
        model.new_migration_rate = model.migration_rate
        # apply a sort of lockdown
        model.new_migration_rate[node, :] -= model.migration_rate[node, :] * restriction
        model.new_migration_rate[:, node] -= model.migration_rate[:, node] * restriction
        model.new_migration_rate[node, node] +=
            (1 - model.migration_rate[node, node]) * restriction
        # normalize the matrix between 0 and 1
        model.new_migration_rate[model.new_migration_rate.<0.0] .= 0.0
        model.new_migration_rate[model.new_migration_rate.>1.0] .= 1.0
    end

    function countermeasures!(model::StandardABM, data::Matrix{Int}, step::Int)
        rate = slope(data[:, (end-step)+1:end])
        # apply countermeasures and update the model
        for i = 1:length(model.η)
            if get_node_status(model, i) > 0.0
                # applico le contromisure solamente se il nodo ha un status > 0
                if rate > 0.0
                    model.η[i] = rate ≥ model.η[i] ? rate : model.η[i]
                    # apply lockdown only if rate is too high
                    # not too sure about this
                    if rate > 0.2
                        apply_lockdown!(model, i, rate)
                    end
                else
                    model.η[i] *= (1.0 + rate)
                end
            end
        end
    end
end

function controller_happiness!(model::StandardABM)
    for i = 1:length(model.η)
        # balance the countermeasures with a simple formula
        h = model.happiness[i]
        model.η[i] =
            h + model.η[i] < model.η[i] / 2 ? model.η[i] * (1 - (model.η[i] / abs(h))) :
            model.η[i]
    end
end

function controller_voc()
    # prova a predire quando uscira' la nuova variante
    # idea molto ambiziosa
end
end
