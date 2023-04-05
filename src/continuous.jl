# link: https://juliadynamics.github.io/AgentsExampleZoo.jl/dev/examples/social_distancing/#Continuous-space-social-distancing
# link interactive: https://juliadynamics.github.io/Agents.jl/stable/agents_visualizations/
# link: https://www.washingtonpost.com/graphics/2020/world/corona-simulator/

module continuous
    using Agents, Random
    # using InteractiveDynamics, GLMakie
    using Plots, LaTeXStrings, StatsPlots
    using DrWatson: @dict

    # posso simulare le interazioni sociali di individui tramite 
    # l'uso di agenti che si comportano come delle palline che rimbalzano.
    # L'idea alla base è quella di simulare le interazioni tra individui 
    # tramite questo modello (che può essere raffinato). quando due palline
    # si scontrano avviene un interazione.

    @agent Person ContinuousAgent{2} begin
        mass::Float64
        days_infected::Int
        immunity::Int
        status::Symbol #:S, :E, :I, :R
        β::Float64
    end

    # valori default esempio
    function model_init(;
        infection_period = 30,
        detection_time = 14,
        exposure_time = 0,
        immunity_period = 365, 
        interaction_radius = 0.012,
        dt = 1.0,
        speed = 0.002,
        death_rate = 0.044, # da WHO
        N = 1000,
        initial_infected = 5,
        βmin = 0.4,
        βmax = 0.8,
        space_dimension = (1.0, 1.0),
        spacing = 0.02,
        steps_per_day = 24,
        seed = 1234,
        )

        # parametri che potrò cambiare con l'interactive window
        properties=(;
            N,
            infection_period,
            detection_time,
            exposure_time,
            immunity_period , 
            interaction_radius,
            death_rate,
            βmin,
            βmax,
            speed,
            dt,
            steps_per_day,
        )
        # cast da variabile immutable a mutable. meglio se struct ad-hoc
        properties = Dict(k=>v for (k,v) in pairs(properties))

        space = ContinuousSpace(space_dimension; spacing=spacing)
        model = ABM(Person, space, properties=properties, rng=Xoshiro(seed))

        # inserisco gli agenti nel modello
        for ind in 1:N
            pos = Tuple(rand(model.rng, 2) .* space_dimension)
            status = ind ≤ N - initial_infected ? :S : :I
            vel = sincos(2π * rand(model.rng)) .* speed
            β = (βmax - βmin) * rand(model.rng) + βmin
            add_agent!(pos, model, vel, 1.0, 0, 0, status, β)
        end
        return model
    end

    function model_step!(model)
        for (a1, a2) in interacting_pairs(model, model.interaction_radius, :nearest)
            transmit!(a1, a2, model)
            elastic_collision!(a1, a2, :mass)
        end
    end

    function agent_step!(agent, model)
        move_agent!(agent, model, model.dt)
        update!(agent, model)
        recover_or_die!(agent, model)
    end

    # funzione per il controllo della trasmissione della malattia
    function transmit!(a1, a2, model)
        count(a.status == :I for a in (a1, a2)) ≠ 1 && return
        infected, healthy = a1.status == :I ? (a1,a2) : (a2,a1)
        n = infected.β * abs(randn(model.rng))
        n ≤ 0 && return
        if healthy.status == :S 
            healthy.status = :E
            n -= 1
            n ≤ 0 && return
        end
    end

    # aggiornamento infetti
    function update!(agent, model) 
        agent.status == :S && return

        if agent.status == :I
            agent.days_infected += 1
        end
        if agent.status == :E
            agent.days_infected += 1
            if agent.days_infected ≥ model.exposure_time 
                agent.status = :I
				agent.days_infected = 1
            end
        end
        agent.immunity -= 1
		if agent.status == :R && agent.immunity ≤ 0
			agent.status = :S
			agent.immunity = 0
		end
    end

    function recover_or_die!(agent, model)
        if agent.days_infected ≥ model.infection_period 
            if rand(model.rng) ≤ model.death_rate
                kill_agent!(agent, model)
            else
                # semplificazione della quarantena
                agent.status = :R
                agent.days_infected = 0
                agent.immunity = model.immunity_period 
            end
        end
    end

    function collect(model; astep = agent_step!, mstep = model_step!, n = 1000)
		susceptible(x) = count(i == :S for i in x)
		exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)

        to_collect = [(:status, f) for f in (susceptible, exposed, infected, recovered, length)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
		return data
	end
	
	function line_plot(data, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered"], title = "ABM Dynamics")
		return @df data plot([data.susceptible_status, data.exposed_status, data.infected_status, data.recovered_status], labels = labels, title = title, lw = 2, xlabel = L"Days")
	end
end