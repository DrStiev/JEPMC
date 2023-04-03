# https://juliadynamics.github.io/Agents.jl/stable/examples/sir/
# modulo per la creazione del modello e definizione dell'agente
module graph_model

	using Agents, Random
	using DrWatson: @dict
	using LinearAlgebra: diagind
	using StatsBase
	using InteractiveDynamics

	include("ode.jl")

	@agent Person GraphAgent begin
		days_infected::Int
		days_of_immunity::Int # dopo che si è guariti si ha un periodo di immunità
		status::Symbol #:S, :E, :I, :R, :Q (:D viene recuperato dal modello)
	end

	function model_init(;
		Ns,
		migration_rates,
		β_und,
		β_det, # infettività status :Q
		infection_period = 30,
		detection_time = 14,
		exposure_time = 5,
		quarantine_time = 14,
		reinfection_probability = 0.05,
		death_rate = 0.02,
		Is = [zeros(Int, length(Ns) - 1)..., 1],
		seed = 0,
		)

		rng = Xoshiro(seed)
		@assert length(Ns) == length(Is) == 
		length(β_und) == length(β_det) ==
		size(migration_rates, 1) "Length of Ns, Is, and B, and number of rows / columns in migration_rates should be the same"
		@assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

		C = length(Ns)
		# normalizzo il migration rates
		migration_rates_sum = sum(migration_rates, dims = 2)
		for c in 1:C
			migration_rates[c, :] ./= migration_rates_sum[c]
		end

		properties = @dict(
			Ns,
			migration_rates,
			β_und,
			β_det,
			infection_period,
			detection_time,
			exposure_time,
			quarantine_time,
			reinfection_probability,
			death_rate,
			Is,
			C	
		)
		space = GraphSpace(Agents.Graphs.complete_graph(C))
		model = ABM(Person, space; properties, rng)

		for city in 1:C, _ in 1:Ns[city]
			add_agent!(city, model, 0, 0, :S) # Suscettibile
		end
		# add infected individuals
		for city in 1:C
			inds = ids_in_position(city, model)
			for n in 1:Is[city]
				agent = model[inds[n]]
				agent.status = :I # Infetto
				agent.days_infected = 1
			end
		end
		return model
	end

	function agent_step!(agent, model)
		migrate!(agent, model)
		transmit!(agent, model)
		update!(agent, model)
		recover_or_die!(agent, model)
	end

	function migrate!(agent, model)
		# se in quarantena non può muoversi
		agent.status == :Q && return
		pid = agent.pos
		m = StatsBase.sample(model.rng, 1:(model.C), StatsBase.Weights(model.migration_rates[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
		end
	end

	function transmit!(agent, model)
		# :S e :E non possono infettare per definizione
		agent.status == :S && return
		agent.status == :E && return

		rate = agent.status == :Q ? model.β_det[agent.pos] : model.β_und[agent.pos]
		n = rate * abs(randn(model.rng))
		n ≤ 0 && return

		for contactID in ids_in_position(agent, model)
			contact = model[contactID]
			if contact.status == :S || 
				(contact.status == :R && 
				contact.days_of_immunity == 0 && 
				rand(model.rng) ≤ model.reinfection_probability)
				contact.status = :E
				n -= 1
				n ≤ 0 && return
			end
		end
	end

	function update!(agent, model)
		if agent.status != :S && agent.status != :R
			agent.days_infected += 1
			if agent.status == :E
				if agent.days_infected ≥ ceil(model.exposure_time[agent.pos])
					agent.status = :I
					agent.days_infected = 1
				end
			end
			if agent.status == :I 
				if agent.days_infected ≥ model.detection_time
					agent.status = :Q
				end
			end
		end
		if agent.days_of_immunity > 0
			agent.days_of_immunity -= 1
		end
	end

	function recover_or_die!(agent, model)
		if agent.days_infected ≥ model.infection_period
			if rand(model.rng) ≤ model.death_rate
				remove_agent!(agent, model)
			else
				agent.status = :R
				agent.days_infected = 0
				agent.days_of_immunity = 30 # random value
			end
		end
	end

	function collect(model; step = agent_step!, n = 100)
		susceptible(x) = count(i == :S for i in x)
		exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        quarantined(x) = count(i == :Q for i in x)

        to_collect = [(:status, f) for f in (susceptible, exposed, infected, recovered, quarantined, length)]
        data, _ = run!(model, step, n; adata = to_collect)
		return data
	end
end