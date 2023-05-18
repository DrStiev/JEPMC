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
		immunity_period::Int
		status::Symbol # :S, :E, :I, :R
		detected::Symbol # :S, :I, :Q, :L, :H, :R, :V
		happiness::Float64 # [-1, 1]
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
		η,  # countermeasures type
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
			R₀, ξ, q, Is, C, is_lockdown=false, β = R₀/γ,
			γ, σ, ω, δ, η, ηₜ = [1, 2, 3, 5, 8, 13, 21, 55, 89, 144], i=0,
		)
		
		# creo il modello 
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
		controls!(model)
		countermeasures!(model)
		lockdown!(model)
		# variant!(model)
	end

	function controls!(model)
		# campiono solamente gli agenti non in quarantena, 
		# in quanto di quelli in :Q conosco già lo stato
		population_sample = sample(model.rng, 
			filter(x -> x.detected ≠ :Q, [agent for agent in allagents(model)]), 
			round(Int, model.ncontrols))
		# number of controls in time
		infected_ratio = length(filter(x -> x == :I, [result!(p, model) for p in population_sample])) / length(population_sample)
		model.ncontrols = infected_ratio ≥ model.infected_ratio ? 
			model.ncontrols*1.2 : model.ncontrols/1.2
		model.infected_ratio = infected_ratio
	end

	function lockdown!(model)
		if false # condizione di attivazione
			model.θₜ = round(Int, abs(rand(Normal(60, 30))))
		end
		if model.θₜ > 0
			# lockdown (proprietà spaziale)
			if model.is_lockdown == false
				model.is_lockdown = true
				population_sample = sample(model.rng, 
					filter(x -> x, [agent for agent in allagents(model)]), 
					round(Int, count(allagents(model))*model.θ))
				for p in population_sample
					p.detected = :L
				end
			end
			model.θₜ -= 1
			model.R₀ -= 0.0014 # random value
		else
			model.is_lockdown = false
		end
	end

	# very very simple function
	function variant!(model)
		# https://www.nature.com/articles/s41579-023-00878-2
		# https://onlinelibrary.wiley.com/doi/10.1002/jmv.27331
		# https://virologyj.biomedcentral.com/articles/10.1186/s12985-022-01951-7
		# nuova variante ogni tot tempo? 
		if rand(model.rng) < 8*10E-4 # condizione di attivazione
			model.R₀ = abs(rand(Normal(model.R₀, model.R₀/10)))
			model.γ = round(Int, abs(rand(Normal(model.γ, model.γ/10))))
			model.σ = round(Int, abs(rand(Normal(model.σ, model.σ/10))))
			model.ω = round(Int, abs(rand(Normal(model.ω, model.ω/10))))
			model.δ = abs(rand(Normal(model.δ, model.δ/10)))
			model.β = model.R₀ / model.γ
		end
	end

	function countermeasures!(model)
		# regole e rateo per i vaccini
		if rand(model.rng) < 1/365 # condizione di attivazione
			model.ξ = abs(rand(Normal(0.0025, 0.00025)))
		end
		# inserisco e tolgo countermeasures
		if model.infected_ratio ≥ 0.01 # condizione di attivazione
			# aumentare le countermeasures via via con il tempo
			model.i += 1
			model.i = model.i > length(model.ηₜ) ? length(model.ηₜ) : model.i
			model.η = 1/model.ηₜ[model.i]
			model.R₀ -= 0.0014 # random value
		else
			model.i -= 1
			model.i = model.i ≤ 0 ? 1 : model.i
			model.η = 1/model.ηₜ[model.i]
			model.R₀ += 0.0014 # random value
		end
	end

	function agent_step!(agent, model)
		# happiness!(agent, -(model.η)^-1/1000, (model.η)^-1/10000)
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

	function happiness!(agent, val, std)
		agent.happiness += rand(Normal(val, std))
		# mantengo la happiness tra [-1, 1]
		agent.happiness = agent.happiness > 1.0 ? 1.0 : agent.happiness < -1.0 ? -1.0 : agent.happiness
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
			happiness!(agent, 0.1, 0.05)
		end
	end

	function transmit!(agent, model)
		agent.status != :I && return
		n = model.β * model.η * abs(randn(model.rng))
		n ≤ 0 && return
		for contactID in ids_in_position(agent, model)
			contact = model[contactID]  
			# se l'agente è in quarantena o è vigente il lockdown
			# non è possibile che venga infettato o infetti
			if contact.detected ≠ :Q && contact.detected ≠ :L && contact.status == :S 
				contact.status = :E 
				n -= 1
				n ≤ 0 && return
			end
		end
	end

	function update_status!(agent, model)
		# fine periodo di latenza
		if agent.status == :E
			if agent.days_infected > model.σ
				agent.status = :I
				agent.days_infected = 1
			end
			agent.days_infected += 1
		# avanzamento malattia
		elseif agent.status == :I
			agent.days_infected += 1
		elseif agent.status == :R
			if rand(model.rng) < 1/agent.immunity_period 
				agent.status = :S
			else
				agent.immunity_period -= 1
			end
		end
	end

	function update_detected!(agent, model)
		# probabilità di vaccinarsi
		if agent.detected == :S
			if rand(model.rng) < model.ξ 
				agent.status = :R
				agent.immunity_period = model.ω
				agent.detected = :V
			end
		# metto in quarantena i pazienti che scopro essere positivi
		elseif agent.detected == :I
			agent.detected = :Q
			agent.days_quarantined = 1
		# avanzamento quarantena
		elseif agent.detected == :Q 
			agent.days_quarantined += 1
			happiness!(agent, -0.05, 0.05)
		elseif agent.detected == :L
			happiness!(agent, -0.05, 0.05)
		end
	end

	function recover_or_die!(agent, model)
		# fine malattia
		if agent.days_infected > model.γ
			# probabilità di morte
			δ = agent.detected == :V ? model.δ / 10 : model.δ
			if rand(model.rng) < δ
				remove_agent!(agent, model)
				return
			end
			# probabilità di guarigione
			agent.status = :R
			agent.days_infected = 0
			agent.immunity_period = model.ω
			variant!(model)
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
		lockdown(x) = count(i == :L for i in x)
		vaccined(x) = count(i == :V for i in x)
		happiness(x) = mean(x)

        to_collect = [(:status, susceptible), (:status, exposed), (:status, infected), 
			(:status, recovered), (:happiness, happiness), (:detected, infected), 
			(:detected, quarantined), (:detected, lockdown), (:detected, recovered), 
			(:detected, vaccined), (:status, dead)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
		data[!, :dead_status] = data[!, end]
    	select!(data, :susceptible_status, :exposed_status, :infected_status, :recovered_status, 
			:infected_detected, :quarantined_detected, :recovered_detected, :lockdown_detected, 
			:vaccined_detected, :dead_status, :happiness_happiness)
        return data
    end
end