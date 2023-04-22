module graph
	using Agents, Random, DataFrames
	using DrWatson: @dict
	using StatsBase: sample, Weights
	using InteractiveDynamics
	using Statistics: mean
	using Distributions

	@agent Person GraphAgent begin
		days_infected::Int
		days_immunity::Int
		days_quarantined::Int
		status::Symbol # :S, :E, :I, :R (:V)
		detected::Symbol # :S, :I, :Q, :R (:V)
		happiness::Float64 # [-1, 1]
	end

	# TODO: parametro che gestisce il lockdown / quarantena
	function init(;
		number_point_of_interest, migration_rate, 
		ncontrols, control_growth, control_accuracy,
		R₀, γ, σ, ω, ξ, δ, η, ϵ, q, θ, T, seed = 1234,
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
		ncontrols *= sum(number_point_of_interest)

		properties = @dict(
			number_point_of_interest, migration_rate, 
			control_accuracy, ncontrols, control_growth, θ,
			R₀, γ, σ, ω, δ, ξ, β = R₀*γ, η, ϵ, q, T, Is, C, 
		)
		
		# creo il modello # Agents.Graphs.clique_graph(C, C ÷ 2)
		model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

		# aggiungo la mia popolazione al modello
		for city in 1:C, _ in 1:number_point_of_interest[city]
			add_agent!(city, model, 0, 0, 0, :S, :S, 0.0) # Suscettibile
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

	function model_step!(model)
		# campiono solamente gli agenti non in quarantena, in quanto di loro conosco lo stato
		population_vector = [agent for agent in allagents(model)]
		population_sample = sample(filter(x -> x.detected ≠ :Q, population_vector), trunc(Int, model.ncontrols))
		for p in population_sample
			result!(p, model)
		end
		# aumento il numero di controlli ad ogni step del modello
		model.ncontrols *= model.control_growth
	end

	function agent_step!(agent, model)
		# θ: variabile lockdown (percentuale)
		if rand(model.rng) > model.θ
			if agent.detected ≠ :Q
				migrate!(agent, model)
			end
		end
		transmit!(agent, model)
		update!(agent, model)
		recover_or_die!(agent, model)
	end	

	function result!(agent, model)
		if agent.status == :I
			agent.detected = rand(model.rng) ≤ model.control_accuracy[1] ? :I : :S
		end
		if agent.status == :E
			agent.detected = rand(model.rng) ≤ model.control_accuracy[2] ? :I : :S
		end
		# in questo caso non ci importa troppo dello stato dell'agente
		# ma ci interessa sapere che non è :I
		if agent.status == :S || agent.status == :R
			agent.detected = rand(model.rng) ≤ model.control_accuracy[3] ? agent.status : :I
		end
		return agent.detected
	end

	function migrate!(agent, model)
		# possibilità di non muoversi per via delle restrizioni
		rand(model.rng) > model.η * 10 && (agent.happiness += rand(Uniform(-0.1, 0.1)))
		pid = agent.pos
		m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
			# control measures could reduce the happiness value
			agent.happiness += rand(Uniform(-0.05, min(model.η, 0.25)))
		end
	end

	function transmit!(agent, model)
		agent.status != :I && return
		for contactID in ids_in_position(agent, model)
			contact = model[contactID]
			if contact.status == :S && rand(model.rng) ≤ (model.β * model.η)
				contact.status = :E 
			end
		end
	end

	function update!(agent, model)
		# probabilità di vaccinarsi
		if agent.detected == :S
			if rand(model.rng) ≤ model.ξ 
				agent.status = :R
				agent.detected = :R
				return
			end
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
		agent.status == :I && (agent.days_infected += 1)
		# perdita progressiva di immunità e aumento rischio exposure
		if agent.status == :R
			agent.days_immunity -= 1
			rand(model.rng) ≤ 1/agent.days_immunity && (agent.status = :S)
		end
		# metto in quarantena i pazienti che scopro essere positivi
		if agent.detected == :I
			agent.detected = :Q
			agent.days_quarantined = 1
			return
		end
		# avanzamento quarantena
		if agent.detected == :Q 
			agent.days_quarantined += 1
			agent.happiness += rand(Uniform(-0.2, 0.05))
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
			rand(model.rng) ≤ 1E-6*(1.0/agent.happiness) && (remove_agent!(agent, model))
		end
		# fine malattia
		if agent.days_infected > 1/model.γ
			# probabilità di morte
			rand(model.rng) ≤ model.δ && (remove_agent!(agent, model))
			# probabilità di guarigione
			agent.status = :R
			agent.days_immunity = 1/model.ω
			agent.days_infected = 0
			return
		end
		if agent.detected == :Q && agent.days_quarantined ≥ 1/model.q
			new_status = result!(agent, model)
			if new_status == :R || new_status == :S
				# agent.happiness += rand(Uniform(0.0, 0.05))
				agent.days_quarantined = 0
			else 
				# lascio il paziente in quarantena per ancora 1/2 del periodo totale
				agent.detected = :Q
				agent.days_quarantined ÷= 2
			end
			return
		end
	end	

	function collect(model, astep, mstep; n = 100)
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        dead(x) = sum(model.number_point_of_interest) - length(x)

		quarantined(x) = count(i == :Q for i in x)
		happiness(x) = mean(x)

        to_collect = [(:status, susceptible), (:status, exposed), (:status, infected), (:status, recovered), (:happiness, happiness), (:detected, infected), (:detected, quarantined), (:status, dead)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
		data[!, :dead_status] = data[!, end]
    	select!(data, :susceptible_status, :exposed_status, :infected_status, :recovered_status, :infected_detected, :quarantined_detected, :dead_status, :happiness_happiness)
        for i in 1:ncol(data)
            data[!, i] = data[!, i] / sum(model.number_point_of_interest)
        end
        return data
    end
end