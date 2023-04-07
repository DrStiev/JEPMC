module graph
	# using OrdinaryDiffEq
	using Agents, Random, StatsBase
	using DrWatson: @dict
	using LinearAlgebra: diagind

	include("ode.jl")

	@agent Person GraphAgent begin
		days_infected::Int
		status::Symbol #:S, :E, :I, :R (:V)
	end

	function init(;
		number_point_of_interest,
		migration_rate,
		β, γ, σ ,ω, α, ϵ, ξ,
		seed = 1234,
		)

		rng = Xoshiro(seed)
		C = length(number_point_of_interest)
		# normalizzo il migration rate
		migration_rate_sum = sum(migration_rate, dims= 2)
		for c in 1:C
			migration_rate[c, :] ./= migration_rate_sum[c]
		end
		# scelgo il punto di interesse che avrà il paziente zero
		Is = [zeros(Int, length(number_point_of_interest) - 1)..., 1]

		properties = @dict(
			number_point_of_interest,
			migration_rate,
			β, γ, σ ,ω, α, ϵ, ξ,
			Is, C
		)
		
		# creo lo spazio per il mio modello
		space = GraphSpace(Agents.Graphs.complete_graph(C))
		# creo il modello
		model = ABM(Person, space; properties, rng)

		# aggiungo la mia popolazione al modello
		for city in 1:C, _ in 1:number_point_of_interest[city]
			add_agent!(city, model, 0, :S) # Suscettibile
		end
		# aggiungo il paziente zero
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
		m = StatsBase.sample(model.rng, 1:(model.C), StatsBase.Weights(model.migration_rate[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
		end
	end

	function transmit!(agent, model)
		agent.status != :I && return
		# number of possible infection from a single agent
		n = model.β * abs(randn(model.rng))
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
		# probabilità di vaccinarsi
		if agent.status == :S
			rand(model.rng) ≤ model.ϵ && (agent.status = :R)
		end
		# fine periodo di latenza
		if agent.status == :E
			if rand(model.rng) ≤ model.ξ 
				agent.status = :S
				agent.days_infected = 0
				return
			end
			if agent.days_infected ≥ (1/model.σ)
				agent.status = :I
				agent.days_infected = 1
				return
			end
			agent.days_infected += 1
		end
		# avanzamento malattia
		agent.status == :I && (agent.days_infected += 1)
		# fine immunità
        if agent.status == :R 
          rand(model.rng) ≤ model.ω && (agent.status = :S)
        end 
	end

	function recover_or_die!(agent, model)
		# fine malattia
		if agent.days_infected ≥ (1 / model.γ)
			# probabilità di morte
			if rand(model.rng) ≤ model.α 
				remove_agent!(agent, model)
			else
				# probabilità di guarigione
				agent.status = :R
				agent.days_infected = 0
			end
		end
	end	 
end