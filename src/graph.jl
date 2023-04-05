# https://juliadynamics.github.io/Agents.jl/stable/examples/sir/
# modulo per la creazione del modello e definizione dell'agente
# https://en.wikipedia.org/wiki/Percolation_theory
# https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model

module graph
	using Agents, Random, StatsBase
	using DrWatson: @dict
	using LinearAlgebra: diagind
	using Plots, LaTeXStrings, StatsPlots
	# using InteractiveDynamics, GraphMakie, GLMakie, Plots

	# include("ode.jl")
	# include("optimizer.jl")

	@agent Person GraphAgent begin
		days_infected::Int
		immunity::Int
		status::Symbol #:S, :E, :I, :R (:D viene recuperato dal modello)
	end

	function model_init(;
		Ns,
		migration_rates,
		β_und,
		β_det,
		infection_period = 30,
		detection_time = 14,
		exposure_time = 0,
		immunity_period = 365,
		death_rate = 0.02,
		Is = [zeros(Int, length(Ns) - 1)..., 1],
		seed = 1234,
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
			immunity_period,
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
		pid = agent.pos
		m = StatsBase.sample(model.rng, 1:(model.C), StatsBase.Weights(model.migration_rates[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
		end
	end

	function transmit!(agent, model)
		# solo :I puo' infettare
		agent.status != :I && return

		rate = agent.days_infected ≥ model.detection_time ? model.β_det[agent.pos] : model.β_und[agent.pos]
		n = rate * abs(randn(model.rng))
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
		agent.status == :S && return

		if agent.status == :I
			agent.days_infected += 1 
		end
		if agent.status == :E
			agent.days_infected += 1 
			if agent.days_infected ≥ model.exposure_time[agent.pos]
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
		if agent.status == :I && 
			agent.days_infected ≥ model.infection_period
			if rand(model.rng) ≤ model.death_rate
				remove_agent!(agent, model)
			else
				agent.status = :R
				agent.days_infected = 0
				agent.immunity = model.immunity_period
			end
		end
	end

	function collect(model; step = agent_step!, n = 1000)
		susceptible(x) = count(i == :S for i in x)
		exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)

        to_collect = [(:status, f) for f in (susceptible, exposed, infected, recovered, length)]
        data, _ = run!(model, step, n; adata = to_collect)
		return data
	end
	
	function line_plot(data, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered"], title = "ABM Dynamics")
		return @df data plot([data.susceptible_status, data.exposed_status, data.infected_status, data.recovered_status], labels = labels, title = title, lw = 2, xlabel = L"Days")
	end
end