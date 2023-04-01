# https://juliadynamics.github.io/Agents.jl/stable/examples/sir/

# modulo per la creazione del modello e definizione dell'agente
module graph_model

	using Agents, Random
	using DrWatson: @dict
	using LinearAlgebra: diagind
	using StatsBase
	using InteractiveDynamics

	@agent Person GraphAgent begin
		days_infected::Int
		vaccine_dose::Int
		status::Symbol #:S, :I, :R, :Q, :V
	end

	function model_init(;
		Ns,
		migration_rates,
		β_und,
		β_det, # infettività status :Q
		social_distancing = false, # diminuisce β_und e β_det
		quarantine = false, 
		vaccine = false, 
		hospital_overwhelmed = false, # aumenta β_det
		mask = false, # diminuisce β_und e β_det
		infection_period = 30,
		detection_time = 14,
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
			social_distancing,
			quarantine,
			vaccine,
			hospital_overwhelmed,
			mask,
			infection_period,
			detection_time,
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

	function create_params(;
		C,
		max_travel_rate,
		min_population = 50,
		max_population = 5000,
		infection_period = 18,
		reinfection_probability = 0.15, # valore moderna 1 dose
		detection_time = 5, 
		quarantine_time = 14,
		death_rate = 0.044, # valore WHO covid
		Is = [zeros(Int, C-1)...,1],
		seed = 19,
		)

		Random.seed!(seed)
		Ns = rand(min_population:max_population, C)
		β_und = rand(0.3:0.2:0.6, C)
		β_det = β_und ./ 10
		social_distancing = quarantine = vaccine = hospital_overwhelmed = mask = false

		Random.seed!(seed)
		migration_rates = zeros(C,C)
		for c in 1:C
			for c2 in 1:C
				migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
			end
		end
		maxM = maximum(migration_rates)
		migration_rates = (migration_rates .* max_travel_rate) ./ maxM
		migration_rates[diagind(migration_rates)] .= 1.0

		params = @dict(
			Ns,
			migration_rates,
			β_und,
			β_det,
			social_distancing,
			quarantine,
			vaccine,
			hospital_overwhelmed,
			mask,
			infection_period,
			detection_time,
			quarantine_time,
			reinfection_probability,
			death_rate,
			Is
		)
		return params
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

	# TODO: possibilità di malcontento oppure troppo difficile da modellare?
	function transmit!(agent, model)
		# se non infetto non può infettare
		agent.status != :I && return
		# aumento infettività se ospedali full aumento rischio solo per β_det
		model.β_det[agent.pos] = model.hospital_overwhelmed ? model.β_det[agent.pos] + (model.β_det[agent.pos] * 0.5) : model.β_det[agent.pos]
		rate = agent.days_infected < model.detection_time ? model.β_und[agent.pos] : model.β_det[agent.pos]
		# riduzione infettività se applico distanziamento sociale
		rate = model.social_distancing ? rate - (rate * 0.8) : rate
		# riduzione infettività se applico mascherine
		rate = model.mask ? rate - (rate * 0.75) : rate
		
		n = rate * abs(randn(model.rng))
		n ≤ 0 && return

		for contactID in ids_in_position(agent, model)
			contact = model[contactID]
			# ugly code
			if contact.status == :S || contact.status == :R || contact.status == :V 
				if contact.status == :R || contact.status == :V
					rp = contact.vaccine_dose > 1 ? 
						model.reinfection_probability / sum(1:sum(1:contact.vaccine_dose)-1) : 
						model.reinfection_probability
					if rand(model.rng) ≤ rp
						contact.status = :I
						n -= 1
						n ≤ 0 && return
					end
				else
					contact.status = :I
					n -= 1
					n ≤ 0 && return
				end
			end
		end
	end

	function update!(agent, model)
		# if !model.hospital_overwhelmed && 
		# 	(count(i.status == :I for i in collect(allagents(model))) ≥ nagents(model) * 0.25)
		# 	InteractiveDynamics.set_value!(model.properties, :hospital_overwhelmed, true)
		# end
		# if model.hospital_overwhelmed &&
		# 	(count(i.status == :I for i in collect(allagents(model))) < nagents(model) * 0.25)
		# 	InteractiveDynamics.set_value!(model.properties, :hospital_overwhelmed, false)
		# end
		if agent.status == :I 
			agent.days_infected +=1
		end
		if model.quarantine && 
			agent.status == :I && 
			agent.days_infected ≥ model.detection_time
			agent.status == :Q
		end
		if model.vaccine 
			# random value
			n = 0.0085 * abs(randn(model.rng))
			n ≤ 0 && return

			# TODO: copertura vaccinale con periodo di attesa
			for contactID in ids_in_position(agent, model)
				contact = model[contactID]
				if (contact.status == :S || contact.status == :R || contact.status == :V) &&
					contact.vaccine_dose < 3
						contact.status = :V
						contact.vaccine_dose += 1
						n -= 1
						n ≤ 0 && return
				end
			end
		end
	end

	function recover_or_die!(agent, model)
		if agent.days_infected ≥ model.infection_period
			if rand(model.rng) ≤ model.death_rate
				remove_agent!(agent, model)
			else
				agent.status = :R
				agent.days_infected = 0
				agent.vaccine_dose += 1
			end
		end
	end

	get_observable(model; agent_step! = agent_step!) = ABMObservable(model; agent_step!)

	# TODO: add run! function that return dataframe?
end
