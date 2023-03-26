# https://juliadynamics.github.io/Agents.jl/stable/examples/sir/
# https://docs.juliahub.com/Agents/nTsV8/3.6.1/examples/sir/
# https://juliadynamics.github.io/Agents.jl/stable/examples/schoolyard/

using Agents, Random
using Distributions: Poisson, DiscreteNonParametric
using InteractiveDynamics
using GLMakie
using DrWatson: @dict
using LinearAlgebra: diagind
using GraphMakie

@agent Person GraphAgent begin
	days_infected::Int
	days_quarantined::Int
	vaccine_dose::Int
	status::Symbol #:S, :I, :R, :V, :Q
end

function model_init(;
	Ns,
	migration_rates,
	β_und,
	β_det,
	social_distancing = false, # diminuisce β_und
	quarantine = false, # diminuisce β_und e β_det
	vaccine = false, # diminuisce β_und e β_det
	hospital_overwhelmed = false, # aumenta β_det
	infection_period = 30,
	reinfection_probability = 0.05,
	detection_time = 14,
	quarantine_time = 14,
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
		Is,
		β_und,
		β_det,
		social_distancing,
		quarantine,
		vaccine,
		hospital_overwhelmed,
		#β_det,
		migration_rates,
		infection_period,
		#infection_period,
		reinfection_probability,
		detection_time,
		quarantine_time,
		C,
		death_rate
	)

	space = GraphSpace(Agents.Graphs.complete_graph(C))
	model = ABM(Person, space; properties, rng)

	for city in 1:C, _ in 1:Ns[city]
		add_agent!(city, model, 0, 0, 0, :S) # Suscettibile
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
	infection_period = 30,
	reinfection_probability = 0.05,
	detection_time = 14, 
	quarantine_time = 14,
	death_rate = 0.02,
	Is = [zeros(Int, C-1)...,1],
	seed = 19,
	)

	Random.seed!(seed)
	Ns = rand(min_population:max_population, C)
	β_und = rand(0.3:0.2:0.6, C)
	β_det = β_und ./ 10
	social_distancing = quarantine = vaccine = hospital_overwhelmed = false

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
		β_und,
		β_det,
		social_distancing,
		quarantine,
		vaccine,
		hospital_overwhelmed,
		migration_rates,
		infection_period,
		reinfection_probability,
		detection_time,
		quarantine_time,
		death_rate,
		Is
	)
	return params
end

function model_step!(model)
	update_params!(model)
end

function update_params!(model)
	# TODO: aggiungi controllore automatico parametri modello
	agent = length(collect(allagents(model)))
	infected = count(a.status == :I for a in collect(allagents(model)))
	recovered = count(a.status == :R for a in collect(allagents(model)))
	vaccinated = count(a.status == :V for a in collect(allagents(model)))

	if infected ≥ agent * 0.01
		InteractiveDynamics.set_value!(model.properties, :social_distancing, true)
	end
	if infected ≥ agent * 0.05
		InteractiveDynamics.set_value!(model.properties, :quarantine, true)
	end
	if !model.vaccine && rand(model.rng) ≤ 0.01
		InteractiveDynamics.set_value!(model.properties, :vaccine, true)
	end
	if recovered + vaccinated ≥ agent * 0.8
		InteractiveDynamics.set_value!(model.properties, :quarantine, false)
		InteractiveDynamics.set_value!(model.properties, :social_distancing, false)
	end
	if infected ≥ agent * 0.8
		InteractiveDynamics.set_value!(model.properties, :hospital_overwhelmed, true)
	end
	if infected ≤ agent * 0.3 && model.hospital_overwhelmed
		InteractiveDynamics.set_value!(model.properties, :hospital_overwhelmed, false)
	end
end

function agent_step!(agent, model)
	quarantine!(agent, model)
	if agent.status != :Q
		migrate!(agent, model)
	end
	transmit!(agent, model)
	vaccine!(agent, model)
	update!(agent, model)
	recover_or_die!(agent, model)
end

function quarantine!(agent, model)
	!model.quarantine && return
	if agent.status == :I && agent.days_infected ≥ model.detection_time
		agent.status = :Q
	end
end

function migrate!(agent, model)
	pid = agent.pos
	d = DiscreteNonParametric(1:(model.C), model.migration_rates[pid, :])
	m = rand(d)
	if m ≠ pid
		move_agent!(agent, m, model)
	end
end

function transmit!(agent, model)
	agent.status == :S && return
	rate = if agent.days_infected < model.detection_time
		model.β_und[agent.pos] = model.social_distancing ? model.β_und[agent.pos] * 0.8 : model.β_und[agent.pos]
		# meno persone per strada
		model.β_und[agent.pos] = agent.status == :Q ? model.β_und[agent.pos] * 0.5 : model.β_und[agent.pos]
	else
		model.β_det[agent.pos] = model.hospital_overwhelmed ? model.β_det[agent.pos] * 1.5 : model.β_det[agent.pos]
		# più difficile contagiare una persona chiusa in casa
		model.β_det[agent.pos] = agent.status == :Q ? model.β_det[agent.pos] * 0.1 : model.β_det[agent.pos]
	end

	n = rate * abs(randn(model.rng))
	n <= 0 && return

	for contactID in ids_in_position(agent, model)
		contact = model[contactID]
		if contact.status == :S || 
			((contact.status == :R || contact.status == :V) && rand(model.rng) ≤ (model.reinfection_probability/(contact.vaccine_dose*2-1)))
			contact.status = :I
			n -= 1
			n <= 0 && return
		end
	end
end

function vaccine!(agent, model)
	!model.vaccine && return
	# TODO: variabile numero di dosi del modello modificabile da controllore
	agent.vaccine_dose ≥ 3 && return
	if agent.status == :S || agent.status == :R || agent.status == :V
		if rand(model.rng) ≤ 0.0085
			agent.vaccine_dose += 1
			agent.status = :V
		end
	end
end

function update!(agent, model) 
	if agent.status == :I
		agent.days_infected += 1
	end
	if agent.status == :Q
		agent.days_infected += 1
	end
end

function recover_or_die!(agent, model)
	if agent.days_infected ≥ model.infection_period
		# se vaccinato meno probabile che muori
		death_rate = agent.vaccine_dose > 0 ? model.death_rate * 0.1 : model.death_rate
		if rand(model.rng) ≤ death_rate
			remove_agent!(agent, model)
		else
			agent.status = :R
			agent.days_infected = 0
			agent.days_quarantined = 0
			agent.vaccine_dose += 1
		end
	end
end

function agent_status()
	susceptible(x) = count(i == :S for i in x)
	infected(x) = count(i == :I for i in x)
	recovered(x) = count(i == :R for i in x)
	vaccinated(x) = count(i == :V for i in x)
	quarantined(x) = count(i == :Q for i in x)
	adata = [(:status, f) for f in (susceptible, infected, recovered, vaccinated, quarantined, length)]
	return adata
end

function plot_model_as_lines(model, agent_step!, model_step!, n)
	adata = agent_status()
	# estremamente lenta all'inizio!
	data, _ = run!(model, agent_step!, model_step!, n; adata = adata) 
	# data[1:10,:]

	N = sum(model.Ns)
	x = data.step
	fig = Figure(resolution = (600, 400))
	ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
	ls = lines!(ax, x, log10.(data[:, aggname(:status, susceptible)]), color = :grey80)
	li = lines!(ax, x, log10.(data[:, aggname(:status, infected)]), color = :red2)
	lr = lines!(ax, x, log10.(data[:, aggname(:status, recovered)]), color = :green)
	lv = lines!(ax, x, log10.(data[:, aggname(:status, vaccinated)]), color = :blue3)
	lq = lines!(ax, x, log10.(data[:, aggname(:status, quarantined)]), color = :burlywood4)
	dead = log10.(N .- data[:, aggname(:status, length)])
	ld = lines!(ax, x, dead, color = :black)
	Legend(fig[1, 2], [ls, li, lr, lv, lq, ld], ["susceptible", "infected", "recovered", "vaccinated", "quarantined", "dead"])
	return fig # non mi convince a pieno 
end

function interactive_graph_plot(model, agent_step!, model_step!)
	adata = agent_status()
	# https://juliadynamics.github.io/Agents.jl/stable/agents_visualizations/#GraphSpace-models-1
	city_size(agent) = 0.005 * length(agent)
	function city_color(agent)
		agent_size = length(agent)
		infected = count(a.status == :I for a in agent)
		recovered = count(a.status == :R for a in agent)
		vaccinated = count(a.status == :V for a in agent)
		quarantined = count(a.status == :Q for a in agent)
		return RGBf((infected + quarantined) / agent_size, recovered / agent_size, vaccinated / agent_size)
	end

	edge_color(model) = fill((:grey, 0.25), Agents.Graphs.ne(model.space.graph))
	function edge_width(model)
		w = zeros(Agents.Graphs.ne(model.space.graph))
		for e in Agents.Graphs.edges(model.space.graph)
			push!(w, 0.004 * length(model.space.stored_ids[e.src]))
			push!(w, 0.004 * length(model.space.stored_ids[e.dst]))
		end
		return w
	end

	graphplotkwargs = (
		layout = GraphMakie.Shell(), # posizione nodi
		arrow_show = true, # mostrare archi orientati
		edge_color = edge_color,
		edge_width = edge_width,
		edge_plottype = :linesegments # needed for tapered edge widths
	)

	# parametri interattivi modello
	params = Dict(
		:infection_period => 1:1:45,
		:detection_time => 1:1:21,
		:quarantine_time => 1:1:45,
	)

	# FIXME: implementare model_step!
	# TODO: vedi abm_model.jl
	fig, abmobs = abmexploration(model;
		agent_step! = agent_step!, 
		model_step! = model_step!,
		params,
		as = city_size, 
		ac = city_color, 
		graphplotkwargs,
		adata,
		alabels = ["Susceptible", "Infected", "Recovered", "Vaccinated", "Quarantined", "Dead"],
	)
	abmobs
	fig # ERROR: Buffer thickness does not have the same length as the other buffers.
	return fig, abmobs
end

params = create_params(
	C = 8,
	min_population = 50,
	max_population = 5000,
	max_travel_rate = 0.01, 
	infection_period = 14, 
	reinfection_probability = 0.15,
	detection_time = 5,
	quarantine_time = 10,
	death_rate = 0.044,
	)
model = model_init(; params...)

fig = plot_model_as_lines(model, agent_step!, model_step!, 1000)
fig

fig, abmobs = interactive_graph_plot(model, agent_step!, model_step!)
abmobs
fig # ERROR: Buffer thickness does not have the same length as the other buffers.