# link: https://juliadynamics.github.io/AgentsExampleZoo.jl/dev/examples/social_distancing/#Continuous-space-social-distancing
# link interactive: https://juliadynamics.github.io/Agents.jl/stable/agents_visualizations/
# link: https://www.washingtonpost.com/graphics/2020/world/corona-simulator/

# using Distributed
# @everywhere begin
using Agents, Random
using InteractiveDynamics
using GLMakie
using DrWatson: @dict
# end
# addprocs(4)

# posso simulare le interazioni sociali di individui tramite 
# l'uso di agenti che si comportano come delle palline che rimbalzano.
# L'idea alla base è quella di simulare le interazioni tra individui 
# tramite questo modello (che può essere raffinato). quando due palline
# si scontrano avviene un interazione.

@agent Person ContinuousAgent{2} begin
    mass::Float64
    days_infected::Int
    days_quarantined::Int
    status::Symbol #:S, :I, :R, :V, :Q
    β::Float64
    dose::Int # valore che indica il numero di vaccini effettuati
    isFrail::Bool # flag che indica se la persona e' fragile 
                # fragile → raddoppia rischio infezione e morte
    noVax::Bool
    noQuarantine::Bool
end

# valori default esempio
function model_init(;
    infection_period = 30,
    detection_time = 14,
    quarantine_time = 0,
    reinfection_probability = 0.05, 
    is_vaccine = 0.0, # true - false
    is_isolated = 0.0, # true - false
    interaction_radius = 0.012,
    dt = 1.0,
    speed = 0.002,
    death_rate = 0.044, # da WHO
    N = 1000,
    initial_infected = 5,
    seed = 42,
    βmin = 0.4,
    βmax = 0.8,
    frail = 0.0, # percentuale
    max_vaccine_per_day = 0.0, # percentuale della popolazione che in un giorno 
                            # e' possibile vaccinare
    space_dimension = (1.0, 1.0),
    spacing = 0.02,
    noVax = 0.0, # percentuale
    noQuarantine = 0.0, # percentuale
    )

    # parametri che potrò cambiare con l'interactive window
    properties=(;
        N,
        infection_period,
        detection_time,
        quarantine_time,
        reinfection_probability,
        interaction_radius,
        is_vaccine,
        is_isolated,
        death_rate,
        βmin,
        βmax,
        frail,
        max_vaccine_per_day,
        speed,
        noVax,
        noQuarantine,
        dt,
    )
    # cast da variabile immutable a mutable. meglio se struct ad-hoc
    properties = Dict(k=>v for (k,v) in pairs(properties))

    space = ContinuousSpace(space_dimension; spacing=spacing)
    model = ABM(Person, space, properties=properties, rng=MersenneTwister(seed))

    # inserisco gli agenti nel modello
    for ind in 1:N
        pos = Tuple(rand(model.rng, 2) .* space_dimension)
        status = ind ≤ N - initial_infected ? :S : :I
        f = rand(model.rng) ≤ frail # individui fragili
        nv = rand(model.rng) ≤ noVax # individui novax
        nq = rand(model.rng) ≤ noQuarantine # individui noquarantine
        vel = sincos(2π * rand(model.rng)) .* speed
        β = (βmax - βmin) * rand(model.rng) + βmin
        add_agent!(pos, model, vel, 1.0, 0, 0, status, β, 0, f, nv, nq)
    end
    return model
end

function model_step!(model)
    # controllore automatico che modifica i parametri di sistema
    update_params!(model)
    r = model.interaction_radius
    for (a1, a2) in interacting_pairs(model, r, :nearest)
        transmit!(a1, a2, model.reinfection_probability)
        elastic_collision!(a1, a2, :mass)
    end
end

# TODO: completare funzione con: automazione numero vaccini per day 
# (aumento in caso di bisogno e riduzione se non)
# FIXME: performance drop
function update_params!(model)
    # ottenere numero di infetti (in relazione al numero di step effettuati)
    infected = count(i.status == :I for i in collect(allagents(model)))
    # ottenere numero di morti (in relazione al numero di step effettuati)
    # dead = model.N - nagents(model)
    # se necessario applicare quarantena
    if infected ≥ nagents(model) * 0.2 && model.is_isolated == 0 # valore arbitrario
        InteractiveDynamics.set_value!(model.properties, :is_isolated, 1) # key = :is_isolated=1/0, :is_vaccine=1
    end
    # lavorare ad un vaccino (aspettare n step)
    # al termine degli n step applicare vaccino 
    if model.is_vaccine == 0 && rand(model.rng) ≤ 0.2 # valore arbitrario per decidere se vaccino is ready
        InteractiveDynamics.set_value!(model.properties, :is_vaccine, 1) # key = :is_isolated=1/0, :is_vaccine=1
    end
    # quando x percentuale della popolazione ha almeno n dosi di vaccino eliminare quarantena
    vaccinated = count(i.status == :V for i in collect(allagents(model)))
    recovered = count(i.status == :V for i in collect(allagents(model)))
    if vaccinated + recovered > nagents(model) * 0.8 && model.is_isolated == 1
        InteractiveDynamics.set_value!(model.properties, :is_isolated, 0)
    end
end

function agent_step!(agent, model)
    # update_params!(model)
    move_agent!(agent, model, model.dt)
    vaccine!(agent, model)
    quarantine!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

# funzione per il controllo della trasmissione della malattia
function transmit!(a1, a2, rp)
    count(a.status == :I for a in (a1, a2)) ≠ 1 && return
    infected, healthy = a1.status == :I ? (a1,a2) : (a2,a1)

    # semplificazione della condizione di fragilita'
    # che aumenta il rischio di infezione (x2)
    βinf = healthy.isFrail ? infected.β * 2 : infected.β
    # semplificazione della quarantena in cui un individuo
    # se in quarantena ha una diminuzione di rischio infezione (/2)
    βinf = healthy.status == :Q ? βinf / 2 : βinf
    rand(model.rng) > βinf && return

    if healthy.status == :R || healthy.status == :V
        # andamento moderna: 1 → 0.15, 2 → 0.05, 3 → 0.01
        rand(model.rng) > (rp / (healthy.dose*2-1)) && return
    end
    healthy.status = :I
    # semplificazione cambio idea da noquarantine a quarantine
    # dopo essere stati esposti al virus
    if agent.noQuarantine
        agent.noQuarantine = rand(model.rng) ≤ 0.5 ? false : true # valore arbitrario
    end
end

# aggiornamento infetti
function update!(agent, model) 
    if agent.status == :I
        agent.days_infected += 1
    elseif agent.status == :Q
        agent.days_infected += 1
        agent.days_quarantined += 1
        # possibilità di infrangere la quarantena
        if rand(model.rng) ≤ 0.2 # valore randomico
            agent.status = :I
            agent.mass = 1.0
            agent.vel = sincos(2π * rand(model.rng)) .* model.speed
        end
    end
end

# possibilità di vaccinarsi
function vaccine!(agent, model)
    # non esiste un vaccino
    model.is_vaccine == 0 && return
    # esiste un vaccino
    if agent.status == :S || agent.status == :R
        # agente novax 
        agent.noVax && return
        # probabilità che una persona dopo aver fatto il covid
        # reputi inutile fare anche il vaccino
        if agent.status == :R
            rand(model.rng) > 0.8 && return # valore arbitrario
        end
        agent.dose ≥ 3 && return # hard cap sul numero di dosi effettuabili
        # semplificazione del massimo numero di vaccini 
        # effettuabili durante un intero giorno di lavoro
        if rand(model.rng) ≤ model.max_vaccine_per_day
            # attuale spike vaccinale dei primi 5 giorni non mi convince
            agent.status = :V
            agent.dose += 1
        end
    end
end

# possibilità di andare in quarantena
function quarantine!(agent, model)
    # non applicata la quarantena
    model.is_isolated == 0 && return
    # applico la quarantena
    if agent.status == :I && agent.days_infected ≥ model.detection_time * steps_per_day
        # agente noquarantine
        agent.noQuarantine && return
        agent.status = :Q
        agent.mass = Inf
        agent.vel = (0.0, 0.0)
    end
end

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period * steps_per_day
        # semplificazione aumento mortalita' tra gli individui fragili
        death = agent.isFrail ? model.death_rate * 2 : model.death_rate
        if rand(model.rng) ≤ death
            kill_agent!(agent, model)
        else
            # semplificazione della quarantena
            agent.status = :R
            agent.dose += 1
            agent.days_infected = 0
            agent.days_quarantined = 0
            agent.mass = 1.0
            agent.vel = sincos(2π * rand(model.rng)) .* model.speed
            # semplificazione conversione noVax → vax su infezione
            if agent.noVax
                agent.noVax = rand(model.rng) ≤ 0.8 ? false : true # valore arbitrario
            end
        end
    end
end

# dizionario con i parametri modificabili
params = Dict(
    :infection_period => 1:1:45,
    :detection_time => 1:1:21,
    :quarantine_time => 1:1:45,
    :interaction_radius => 0:0.001:1,
    # variabili controllate autonomamente dal modello
    # :is_vaccine => 0:1:1,
    # :is_isolated => 0:1:1,
    :death_rate => 0:0.001:1,
    :max_vaccine_per_day => 0:0.0001:0.1,
    # i parametri commentati sono parametri attualmente
    # non utilizzati nell'interactive window
    #:reinfection_probability => 0:0.01:1,
    #:βmin => 0:0.01:1,
    #:βmax => 0:0.01:1,
    #:frail => 0:0.01:1,
    #:noVax => 0:0.01:1,
    #:noQuarantine => 0:0.01:1,
    #:N => 100:100:10_000,
)

# monitoro i valori del modello 
colors(a) = a.status == :S ? "cornsilk4" : a.status == :I ? "red" : a.status == :V ? "blue" : a.status == :Q ? "brown" : a.status == :R ? "green" : "black"

susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
vaccinated(x) = count(i == :V for i in x)
quarantined(x) = count(i == :Q for i in x)
recovered(x) = count(i == :R for i in x)

dead(model) = model.N - nagents(model) # questo dato lo ottego dal modello non dall'agente!

adata = [(:status, susceptible), (:status, infected), (:status, vaccinated), (:status, quarantined), (:status, recovered)]
mdata = [dead]
plotkwargs = (; ac = colors)

# 24 step equivalgono a 1 giorno.
const steps_per_day = 24

standard_params = (;
    infection_period = 14, # valore arbitrario
    detection_time = 5, # valore arbitrario
    quarantine_time = 10, # valore arbitrario
    reinfection_probability = 0.15, # 1 dose moderna
    is_vaccine = 0, # true - false
    is_isolated = 0, # true - false
    interaction_radius = 0.012, # valore arbitrario
    dt = 1.0,
    speed = 0.002,
    death_rate = 0.044, # da WHO
    N = 500,
    initial_infected = 5,
    seed = 123,
    βmin = 0.1,
    βmax = 0.8,
    frail = 0.01, # valore arbitrario
    max_vaccine_per_day = 0.0085, # valore arbitrario 500k/59M
    space_dimension = (1.0, 1.0),
    spacing = 0.02,
    noVax = 0.1, # valore arbitrario
    noQuarantine = 0.05, # valore arbitrario
)

model = model_init(;standard_params...)

fig, abmobs = abmexploration(model;
    agent_step! = agent_step!, 
    model_step! = model_step!, 
    params, 
    plotkwargs...,
    adata, 
    alabels = ["Susceptible", "Infected", "Vaccinated", "Quarantined", "Recovered"],
    mdata,
    mlabels = ["Dead"]
)
abmobs
fig