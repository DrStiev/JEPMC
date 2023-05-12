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
		β::Float64 # agent infectiveness
		γ::Int # agent recovery time
		σ::Int # agent exposure time
		ω::Int # immunity period
		δ::Float64 # mortality rate
	end

	function init(;
		number_point_of_interest, migration_rate, 
		ncontrols, control_growth, control_accuracy,
		R₀, # R₀ 
		γ,  # periodo infettivita'
		σ,  # periodo esposizione
		ω,  # periodo immunita
		ξ,  # 1 / vaccinazione per milion per day
		δ,  # mortality rate
		η,  # 1 / countermeasures
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
		ncontrols *= sum(number_point_of_interest)
	
		properties = @dict(
			number_point_of_interest, migration_rate, θₜ,
			control_accuracy, ncontrols, control_growth, θ,
			R₀, ξ, η, q, Is, C, qinit = q, is_lockdown=false,
		)
		
		# creo il modello 
		model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

		# aggiungo la mia popolazione al modello
		for city in 1:C, _ in 1:number_point_of_interest[city]
			# assumo μ = valore medio dati, σ = μ/10
			γₐ = round(Int, abs(rand(Normal(γ, γ/10))))
			σₐ = round(Int, abs(rand(Normal(σ, σ/10))))
			δₐ = abs(rand(Normal(δ, δ/10)))
			ωₐ = round(Int, abs(rand(Normal(ω, ω/10))))
			add_agent!(city, model, 0, 0, 0, :S, :S, 0.0, 0.0, γₐ, σₐ, ωₐ, δₐ) # Suscettibile
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

	function variant!(β, γ, σ, ω, δ)
		# https://www.nature.com/articles/s41579-023-00878-2
		# 240 giorni prima che iniziasse ad essere rilevata una
		# mutazione (round(Int, abs(Normal(time, time/10))))
		# nuova variante: modifica β, γ, σ, ω e δ tramite una normale
	end

	function model_step!(model)
		# campiono solamente gli agenti non in quarantena, 
		# in quanto di quelli in :Q conosco già lo stato
		population_sample = sample(model.rng, 
			filter(x -> x.detected ≠ :Q, [agent for agent in allagents(model)]), 
			round(Int, model.ncontrols))
		for p in population_sample
			result!(p, model)
		end
		# ad ogni passo incremento il numero di controlli effettuati
		# questo incremento potrebbe seguire una curva invece che 
		# essere fissato
		model.ncontrols += model.ncontrols*model.control_growth
		if model.θₜ > 0
			# lockdown (proprietà spaziale)
			if model.θ > 0 && model.is_lockdown == false
				model.is_lockdown = true
				model.q = model.θₜ
				population_sample = sample(model.rng, 
					filter(x -> x, [agent for agent in allagents(model)]), 
					round(Int, count(allagents(model))*model.θ))
				for p in population_sample
					p.β = 0.0
					p.detected = :Q
				end
			end
			model.θₜ -= 1
		else
			model.q = model.qinit
			model.is_lockdown = false
		end
		# model.θₜ > 0 && (model.θₜ -= 1)
	end

	function agent_step!(agent, model)
		# mantengo la happiness tra [-1, 1]
		agent.happiness = agent.happiness > 1.0 ? 1.0 : agent.happiness < -1.0 ? -1.0 : agent.happiness
		# θ: variabile lockdown (percentuale)
		if rand(model.rng) < model.θ && model.θₜ > 0
			agent.happiness += rand(Normal(-0.1, 0.05))
		else
			if agent.detected ≠ :Q
				# possibilità di migrare e infettare sse non in quarantena
				migrate!(agent, model)
				transmit!(agent, model)
			end
		end
		update_status!(agent, model)
		update_detection!(agent, model)
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

	function transmit!(agent, model)
		agent.status != :I && return
		for contactID in ids_in_position(agent, model)
			contact = model[contactID]  
			# assunzione stravagante sul lockdown
			lock = model.θₜ > 0 ? (1.0-model.θ) : 1
			if contact.status == :S && rand(model.rng) < (agent.β * model.η * lock)
				contact.status = :E 
				contact.β = abs(rand(Normal(model.R₀, model.R₀/10)))/contact.γ
			end
		end
	end

	function update_status!(agent, model)
		# fine periodo di latenza
		if agent.status == :E
			agent.days_infected += 1
			if agent.days_infected > agent.σ
				agent.status = :I
				agent.days_infected = 1
			end
		# avanzamento malattia
		elseif agent.status == :I
			agent.days_infected += 1
		# perdita progressiva di immunità e aumento rischio exposure
		elseif agent.status == :R
			# agent.days_immunity -= 1
			# non sembra comportarsi come dovrebbe
			if rand(model.rng) < 1/agent.ω # 1/agent.days_immunity 
				agent.status = :S
				agent.days_infected = 0
				agent.days_immunity = 0
			end
		end
	end

	function update_detection!(agent, model)
		# probabilità di vaccinarsi
		if agent.detected == :S
			if rand(model.rng) < model.ξ 
				agent.status = :R
				agent.detected = :R
				agent.days_immunity = agent.ω
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
			agent.happiness += rand(Normal(0, 0.05))
			# troppa o troppo poca felicita' possono portare problemi
			if rand(model.rng) > 1-abs(agent.happiness) 
				# riprende ad essere infettivo
				agent.β = abs(rand(Normal(model.R₀, model.R₀/10)))/agent.γ
				migrate!(agent, model)
				transmit!(agent, model)
			end
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
			# agent.days_immunity = agent.ω
			agent.days_infected = 0
			agent.β = 0.0
		end
	end	

	function exit_quarantine!(agent, model)
		if agent.detected == :Q && agent.days_quarantined > model.q 
			if result!(agent, model) == :S
				agent.days_quarantined = 0
				agent.detected = :R
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