module graph
	using Agents, Random, DataFrames
	using DrWatson: @dict
	using StatsBase: sample, Weights
	using InteractiveDynamics
	using Statistics: mean

	@agent Person GraphAgent begin
		days_infected::Int
		days_immunity::Int
		status::Symbol #:S, :E, :I, :Q, :R (:V)
		happiness::Float64 # [-1, 1]
	end

	# TODO: parametro che gestisce il lockdown / quarantena
	function init(;
		number_point_of_interest, migration_rate,
		R₀, γ, σ, ω, ξ, δ, η, ϵ, T, seed = 1234,
		)
		rng = Xoshiro(seed)
		C = length(number_point_of_interest)
		# normalizzo il migration rate
		migration_rate_sum = sum(migration_rate, dims=2)
		for c in 1:C
			migration_rate[c, :] ./= migration_rate_sum[c]
		end
		# scelgo il punto di interesse che avrà il paziente zero
		Is = [zeros(Int, length(number_point_of_interest) - 1)..., 1]

		properties = @dict(
			number_point_of_interest, migration_rate,
			R₀, γ, σ, ω, δ, ξ, β = R₀*γ, η, ϵ, T, Is, C
		)
		
		# creo il modello
		model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

		# aggiungo la mia popolazione al modello
		for city in 1:C, _ in 1:number_point_of_interest[city]
			add_agent!(city, model, 0, 0, :S, 0.0) # Suscettibile
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

	# la struttura con :Q non mi convince troppo in vista di un futuro controller, but still
	function agent_step!(agent, model)
		agent.status != :Q && migrate!(agent, model)
		transmit!(agent, model)
		update!(agent, model)
		recover_or_die!(agent, model)
	end

	function migrate!(agent, model)
		pid = agent.pos
		m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
			# control measures could reduce the happiness value
			agent.happiness += rand(Uniform(0.0, min(model.η, 0.2)))
		end
	end

	function transmit!(agent, model)
		agent.status != :I && return
		for contactID in ids_in_position(agent, model)
			contact = model[contactID]
			# @show contact
			# la curva sembra troppo ripida
			if contact.status == :S && (rand(model.rng) ≤ (model.β * model.η))
				contact.status = :E 
			end
		end
	end

	function update!(agent, model)
		# probabilità di vaccinarsi
		if agent.status == :S
			rand(model.rng) ≤ model.ξ && (agent.status = :R)
		end
		# fine periodo di latenza
		if agent.status == :E
			rand(model.rng) ≤ model.ϵ && (agent.status = :S)
			if agent.days_infected ≥ (1/model.σ)
				agent.status = :I
				agent.days_infected = 1
				return
			end
			agent.days_infected += 1
		end
		# avanzamento malattia + possibilità di andare in quarantena
		if agent.status == :I
			agent.days_infected += 1
			# TODO: validare giorni di latenza quarantena
			if agent.days_infected > rand(Uniform(1, 5))
				rand(model.rng) < 0.1 && (agent.status = :Q)
			end
		end
		# quarantena paziente dipendente da fattori esterni
		# TODO: inserire durata quarantena come parametro es. θ
		if agent.status == :Q
			agent.days_infected += 1
			agent.happiness += rand(Uniform(-0.1, 0.1))
			return
		end
		# perdita progressiva di immunità e aumento rischio exposure
		if agent.status == :R
			agent.days_immunity -= 1
			rand(model.rng) ≤ 1/agent.days_immunity && (agent.status = :S)
		end
		# se si è molto felici si tende a voler fare di più. questo può portare ad 
		# infrangere la quarantena uscendo anche quando non si dovrebbe
		if agent.happiness ≥ 0.9
			agent.happiness ≥ 1.0 && (migrate!(agent, model))
			rand(model.rng) > agent.happiness && (migrate!(agent, model))
		end
	end

	function recover_or_die!(agent, model)
		# depressione + suicidio
		if agent.happiness ≤ -1.0
			rand(model.rng) ≤ 1E-6 && (remove_agent!(agent, model))
		end
		# fine malattia
		if agent.days_infected > 1/model.γ
			# probabilità di morte
			rand(model.rng) ≤ model.δ && (remove_agent!(agent, model))
			# probabilità di guarigione
			agent.status = :R
			agent.happiness += rand(Uniform(0.0, 0.05))
			agent.days_immunity = 1/model.ω
			agent.days_infected = 0
		end
	end	

	function collect(model, astep; n = 100)
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
		quarantined(x) = count(i == :Q for i in x)
        recovered(x) = count(i == :R for i in x)
        dead(x) = sum(model.number_point_of_interest) - length(x)
		happiness(x) = mean(x)

        to_collect = [(:happiness, happiness), (:status, susceptible), (:status, exposed), (:status, infected), (:status, quarantined), (:status, recovered), (:status, dead)]
        data, _ = run!(model, astep, n; adata = to_collect)
		data[!, :dead_status] = data[!, end]
    	select!(data, :susceptible_status, :exposed_status, :infected_status, :quarantined_status, :recovered_status, :dead_status, :happiness_happiness)
        for i in 1:ncol(data)
            data[!, i] = data[!, i] / sum(model.number_point_of_interest)
        end
        return data
    end
end