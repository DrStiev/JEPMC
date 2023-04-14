module graph
	using Agents, Random, DataFrames
	using DrWatson: @dict
	using StatsBase: sample, Weights

	include("ode.jl")

	# TODO: add quarantine status, where people cannot migrate
	# TODO: add mitigation with η (← \eta) (e.g social distancing, masks, etc...)
	# η describes the control rate, or the speed at which restrictions are imposed
	# TODO: add lockdown where none can migrate 
	@agent Person GraphAgent begin
		days_infected::Int
		status::Symbol #:S, :E, :I, :R (:V)
	end

	# TODO: add ODE
	function init(;
		number_point_of_interest,
		migration_rate, params, 
		seed = 1234,
		)
		(;R₀_n, R̅₀, γ, σ, ψ, η, ξ, θ, δ₀, ω, ϵ) = params
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

		prob = ode.get_ODE_problem(ode.F, [(sumNPOI-1)/sumNPOI, 0.0, 1.0/sumNPOI, 0.0, 0.0, R₀_n, δ₀], (0.0, Inf), params)
		integrator = ode.get_integrator(prob)

		properties = @dict(
			number_point_of_interest,
			migration_rate,
			R₀ = R₀_n, γ, σ ,ω, δ = δ₀, ϵ, β = R₀_n*γ,
			Is, C, integrator, sol = integrator.u[1:5]
		)
		
		# creo il modello
		model = ABM(Person, GraphSpace(Agents.Graphs.complete_graph(C)); properties, rng)

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

	function model_step!(model)
		# effect 1 step
		ode.make_step!(model.integrator, 1.0, true)
		# save the partial solution
		model.sol = model.integrator.u[1:5]
		# notify the integrator that conditions may be altered
		ode.notify_change_u!(model.integrator, true)
	end

	function agent_step!(agent, model)
		migrate!(agent, model)
		# transmit!(agent, model)
		# update!(agent, model)
		# recover_or_die!(agent, model)
	end

	function migrate!(agent, model)
		pid = agent.pos
		m = sample(model.rng, 1:(model.C), Weights(model.migration_rate[pid, :]))
		if m ≠ pid
			move_agent!(agent, m, model)
		end
	end

	function transmit!(agent, model)
		agent.status != :I && return
		# number of possible infection from a single agent
		n = model.β * abs(randn(model.rng))
		# n ≤ 0 && return

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
			rand(model.rng) ≤ model.δ && (remove_agent!(agent, model))
			# probabilità di guarigione
			agent.status = :R
			agent.days_infected = 0
		end
	end	
	
	function collect(model, astep, mstep, t)
		_, sol = run!(model, astep, mstep, t; mdata = [:sol])
		df = DataFrame([getindex.(sol[!,2], i) for i in 1:5], [:susceptible, :exposed, :infected, :recovered, :dead])
		return df
	end

	# function collect(model, astep, mstep; n = 100)
    #     susceptible(x) = count(i == :S for i in x)
    #     exposed(x) = count(i == :E for i in x)
    #     infected(x) = count(i == :I for i in x)
    #     recovered(x) = count(i == :R for i in x)
    #     dead(x) = sum(model.number_point_of_interest) - length(x)

    #     to_collect = [(:status, f) for f in (susceptible, exposed, infected, recovered, dead)]
    #     data, _ = run!(model, astep, mstep, n; adata = to_collect)
    #     data[!, :dead_status] = data[!, 6]
    #     select!(data, :susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead_status)
    #     for i in 1:5
    #         data[!, i] = data[!, i] / sum(model.number_point_of_interest)
    #     end
    #     return data
    # end
end