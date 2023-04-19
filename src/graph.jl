module graph
	using Agents, Random, DataFrames
	using DrWatson: @dict
	using StatsBase: sample, Weights
	using InteractiveDynamics

	# FIXME: mettimi a posto una volta capito come far funzionare il codice di rackauckas
	# include("uode.jl")

	@agent Person GraphAgent begin
		days_infected::Int
		status::Symbol #:S, :E, :I, :R (:V)
		happiness::Float64 # [-1, 1]
	end

	function init(;
		number_point_of_interest, migration_rate,
		R₀, γ, σ, ω, ξ, δ, η, T, seed = 1234,
		)
		rng = Xoshiro(seed)
		C = length(number_point_of_interest)
		sumNPOI = sum(number_point_of_interest)
		# normalizzo il migration rate
		migration_rate_sum = sum(migration_rate, dims=2)
		for c in 1:C
			migration_rate[c, :] ./= migration_rate_sum[c]
		end
		# scelgo il punto di interesse che avrà il paziente zero
		Is = [zeros(Int, length(number_point_of_interest) - 1)..., 1]

		# prob = ode.get_ODE_problem(ode.SEIR, 
		# 	[(sumNPOI-1)/sumNPOI, 0.0, 1.0/sumNPOI, 0.0, 0.0], (0.0, Inf), 
		# 	[R₀, γ, σ, ω, ξ, δ, η])
		# integrator = ode.get_integrator(prob)

		properties = @dict(
			number_point_of_interest, migration_rate,
			R₀, γ, σ, ω, δ, ξ, β = R₀*γ, η, T,
			Is, C #, integrator, situation = integrator.u[1:5]
		)
		
		# creo il modello
		model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

		# aggiungo la mia popolazione al modello
		for city in 1:C, _ in 1:number_point_of_interest[city]
			add_agent!(city, model, 0, :S, 0.0) # Suscettibile
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
		# # effect 1 step
		# ode.make_step!(model.integrator, 1.0, true)
		# # save the partial solution
		# model.situation = model.integrator.u[1:5]
		# # notify the integrator that conditions may be altered
		# ode.notify_change_u!(model.integrator, true)
	end

	function agent_step!(agent, model)
		if agent.status != :Q
			migrate!(agent, model)
		end
		transmit!(agent, model)
		update!(agent, model)
		recover_or_die!(agent, model)
	end

	function migrate!(agent, model)
		pid = agent.pos
		m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
			# cotrol measures could reduce the happiness value
			agent.happiness += rand(Uniform(-0.01, min(model.η, 0.02)))
		end
	end

	function transmit!(agent, model)
		agent.status != :I && return
		# number of possible infection from a single agent
		# with the application of countermeasures
		# n = model.β * model.η * abs(randn(model.rng))
		# n ≤ 0 && return

		for contactID in ids_in_position(agent, model)
			contact = model[contactID]
			if contact.status == :S && rand(model.rng) ≤ (model.β * model.η) # || 
				# (contact.status == :R && rand(model.rng) ≤ model.ω)
				contact.status = :E 
				# n -= 1
				# n ≤ 0 && return
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
			rand(model.rng) ≤ model.σ && (agent.status = :I)
			if agent.days_infected ≥ (1/model.σ)
				agent.status = :I
				agent.days_infected = 1
				return
			end
			agent.days_infected += 1
		end
		# avanzamento malattia + possibilità di andare in quarantena
		# agent.status == :I && (agent.days_infected += 1)
		if agent.status == :I
			agent.days_infected += 1
			# TODO: validare giorni di latenza quarantena
			if agent.days_infected > rand(Uniform(1, 5))
				rand(model.rng) < 0.3 && (agent.status = :Q)
			end
		end
		# quarantena paziente
		if agent.status == :Q
			agent.days_infected += 1
			agent.happiness += rand(Uniform(-0.025, 0.01))
			return
		end
		# perdita immunità
		if agent.status == :R
			rand(model.rng) ≤ model.ω && (agent.status = :S)
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
		if agent.days_infected ≥ (1 / model.γ)
			if rand(model.rng) ≤ model.γ
				# probabilità di morte
				rand(model.rng) ≤ model.δ && (remove_agent!(agent, model))
				# probabilità di guarigione
				agent.status = :R
				agent.happiness += rand(Uniform(-0.01, 0.01))
				# agent.days_infected = 0
			end
		end
	end	
	
	# function collect(model, astep, mstep, t)
	# 	_, sol = run!(model, astep, mstep, t; mdata = [:situation])
	# 	df = DataFrame([getindex.(sol[!,2], i) for i in 1:5], [:susceptible, :exposed, :infected, :recovered, :dead])
	# 	return df
	# end

	function collect(model, astep; n = 100)
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
		quarantined(x) = count(i == :Q for i in x)
        recovered(x) = count(i == :R for i in x)
        dead(x) = sum(model.number_point_of_interest) - length(x)

        to_collect = [(:status, f) for f in (susceptible, exposed, infected, quarantined, recovered, dead)]
        data, _ = run!(model, astep, n; adata = to_collect)
		data[!, :dead_status] = data[!, 7]
    	select!(data, :susceptible_status, :exposed_status, :infected_status, :quarantined_status, :recovered_status, :dead_status)
        for i in 1:6
            data[!, i] = data[!, i] / sum(model.number_point_of_interest)
        end
        return data
    end

	get_observable(model) = ABMObservable(model; agent_step!, model_step!)
end