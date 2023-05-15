module graph
	using Agents, Random, DataFrames
	using DrWatson: @dict
	using StatsBase: sample, Weights
	using InteractiveDynamics
	using Statistics: mean
	using Distributions

	@agent Person GraphAgent begin
		days_infected::Int
		days_quarantined::Int
		status::Symbol # :S, :E, :I, :R (:V)
		detected::Symbol # :S, :I, :Q (:L), :R (:V)
		happiness::Float64 # [-1, 1]
		β::Float64 # agent infectiveness
		γ::Int # agent recovery time
		σ::Int # agent exposure time
		ω::Int # immunity period
		δ::Float64 # mortality rate
		η::Float64 # reduction from using countermeasures
	end

	function init(;
		number_point_of_interest, migration_rate, 
		ncontrols, control_accuracy,
		R₀, # R₀ 
		γ,  # periodo infettivita'
		σ,  # periodo esposizione
		ω,  # periodo immunita
		ξ,  # 1 / vaccinazione per milion per day
		δ,  # mortality rate
		η,  # 1 / countermeasures [0.0:1.0]
		q,  # periodo quarantena
		θ,  # percentage of people under full lockdown
		θₜ, # duration of lockdown ≥ 0
		seed = 1337,
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
		ncontrols = round(Int, ncontrols*sum(number_point_of_interest))+1
	
		properties = @dict(
			number_point_of_interest, migration_rate, θₜ,
			control_accuracy, ncontrols, θ, infected_ratio=0.0,
			R₀, ξ, q, Is, C, qinit = q, is_lockdown=false,
			γ, σ, ω, δ, η,
		)
		
		# creo il modello 
		model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

		# aggiungo la mia popolazione al modello
		for city in 1:C, _ in 1:number_point_of_interest[city]
			ηₐ = abs(rand(Normal(η, η/10))) # semplicistico
			ηₐ = ηₐ > 1.0 ? 1.0 : ηₐ # correzione valore out of bounds
			add_agent!(city, model, 0, 0, :S, :S, 0.0, 0.0, 0.0, 0, 0, 0.0, ηₐ) # Suscettibile
		end
		# aggiungo il paziente zero
		for city in 1:C
			inds = ids_in_position(city, model)
			for n in 1:Is[city]
				agent = model[inds[n]]
				agent.status = :I # Infetto
				agent.days_infected = 1
				agent.β = abs(rand(Normal(R₀, R₀/10)))/agent.γ
			end
		end
		return model
	end

	function model_step!(model)
		# campiono solamente gli agenti non in quarantena, 
		# in quanto di quelli in :Q conosco già lo stato
		population_sample = sample(model.rng, 
			filter(x -> x.detected ≠ :Q, [agent for agent in allagents(model)]), 
			model.ncontrols)
		
		# number of controls in time
		infected_ratio = length(filter(x -> x == :I, [result!(p, model) for p in population_sample])) / length(population_sample)
		if infected_ratio > model.infected_ratio
			model.ncontrols *= 2 # simple growth
			model.infected_ratio = infected_ratio
		else
			model.ncontrols = round(Int, model.ncontrols*0.5)+1
			model.infected_ratio = infected_ratio
		end

		if model.θₜ > 0
			# lockdown (proprietà spaziale)
			if model.θ > 0 && model.is_lockdown == false
				model.is_lockdown = true
				# model.q = model.θₜ
				population_sample = sample(model.rng, 
					filter(x -> x, [agent for agent in allagents(model)]), 
					round(Int, count(allagents(model))*model.θ))
				for p in population_sample
					p.β = 0.0
					p.detected = :L
				end
			end
			model.θₜ -= 1
		else
			# model.q = model.qinit
			model.is_lockdown = false
		end
	end

	function agent_step!(agent, model)
		# mantengo la happiness tra [-1, 1]
		# piu' contromisure ci sono piu' infelici saranno i miei agenti
		agent.happiness += rand(Normal(-(agent.η)^-1/1000, (agent.η)^-1/1000))
		agent.happiness = agent.happiness > 1.0 ? 1.0 : agent.happiness < -1.0 ? -1.0 : agent.happiness
		# troppa o troppo poca felicita' possono portare problemi
		if rand(model.rng) > 1-abs(agent.happiness) 
			migrate!(agent, model)
			transmit!(agent, model)
		end
		if agent.detected ≠ :Q && agent.detected ≠ :L
			# possibilità di migrare e infettare sse non in quarantena
			migrate!(agent, model)
			transmit!(agent, model)
		end
		update_status!(agent, model)
		update_detected!(agent, model)
		recover_or_die!(agent, model)
		exit_quarantine!(agent, model)
	end	

	function result!(agent, model)
		if agent.status == :I
			agent.detected = rand(model.rng) < model.control_accuracy[1] ? :I : :S
		elseif agent.status == :E
			agent.detected = rand(model.rng) < model.control_accuracy[2] ? :I : :S
		else 
			agent.detected = rand(model.rng) < model.control_accuracy[3] ? :S : :I
		end
		return agent.detected
	end

	function migrate!(agent, model)
		pid = agent.pos
		m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
			agent.happiness += rand(Normal(0.1, 0.05))
		end
	end

	function variant!()
		# https://www.nature.com/articles/s41579-023-00878-2
		# nuova variante ogni tot tempo?
	end

	function transmit!(agent, model)
		agent.status != :I && return
		for contactID in ids_in_position(agent, model)
			contact = model[contactID]  
			# se l'agente è in quarantena o è vigente il lockdown
			# non è possibile che venga infettato o infetti
			if contact.detected ≠ :Q && contact.detected ≠ :L
				if (contact.status == :S ||
					(contact.status == :R && rand(model.rng) < 1/contact.ω)) && 
					rand(model.rng) < (agent.β * agent.η * contact.η)
					contact.status = :E 
					# assumo μ = valore medio dati, σ = μ/10
					contact.γ = round(Int, abs(rand(Normal(model.γ, model.γ/10))))
					contact.β = abs(rand(Normal(model.R₀/contact.γ, model.R₀/contact.γ/10)))
					contact.σ = round(Int, abs(rand(Normal(model.σ, model.σ/10))))
					contact.δ = abs(rand(Normal(model.δ, model.δ/10)))
				end
			end
		end
	end

	function update_status!(agent, model)
		# fine periodo di latenza
		if agent.status == :E
			if agent.days_infected > agent.σ
				agent.status = :I
				agent.days_infected = 1
			end
			agent.days_infected += 1
		# avanzamento malattia
		elseif agent.status == :I
			agent.days_infected += 1
		# perdita progressiva di immunità e aumento rischio exposure
		elseif agent.status == :R
			if rand(model.rng) < 1/agent.ω 
				agent.status = :S
			end
		end
	end

	function update_detected!(agent, model)
		# probabilità di vaccinarsi
		if agent.detected == :S
			if rand(model.rng) < model.ξ 
				agent.status = :R
				agent.detected = :R
				agent.ω = round(Int, abs(rand(Normal(model.ω, model.ω/10))))
			end
		# metto in quarantena i pazienti che scopro essere positivi
		elseif agent.detected == :I
			agent.detected = :Q
			agent.β = 0.0 # riduzione infettività
			agent.days_quarantined = 1
		# avanzamento quarantena
		elseif agent.detected == :Q 
			agent.β = 0.0
			agent.days_quarantined += 1
			agent.happiness += rand(Normal(-0.05, 0.05))
		elseif agent.detected == :L
			agent.happiness += rand(Normal(-0.05, 0.05))
		end
	end

	function recover_or_die!(agent, model)
		# fine malattia
		if agent.days_infected > agent.γ
			# probabilità di morte
			if rand(model.rng) < agent.δ
				remove_agent!(agent, model)
				return
			end
			# probabilità di guarigione
			agent.status = :R
			agent.days_infected = 0
			agent.β = 0.0
			agent.ω = round(Int, abs(rand(Normal(model.ω, model.ω/10))))
		end
	end	

	function exit_quarantine!(agent, model)
		if agent.detected == :Q && agent.days_quarantined > model.q 
			if result!(agent, model) == :S
				agent.days_quarantined = 0
				if agent.detected ≠ :L || (agent.detected == :L && model.θₜ ≤ 0)
					agent.detected = :R
				else
					agent.detected = :L
				end
			else 
				# prolungo la quarantena
				agent.days_quarantined ÷= 2
			end
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

        to_collect = [(:status, susceptible), (:status, exposed), (:status, infected), 
			(:status, recovered), (:happiness, happiness), (:detected, infected), 
			(:detected, quarantined), (:detected, recovered), (:status, dead)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
		data[!, :dead_status] = data[!, end]
    	select!(data, :susceptible_status, :exposed_status, :infected_status, :recovered_status, 
			:infected_detected, :quarantined_detected, :recovered_detected, 
			:dead_status, :happiness_happiness)
        return data
    end
end