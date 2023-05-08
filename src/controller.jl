module controller
	using DataFrames, DataDrivenDiffEq, DataDrivenSparse
	using LinearAlgebra, OrdinaryDiffEq, ModelingToolkit
	using Plots, Random
	include("graph.jl")
	include("params.jl")
	include("uode.jl")

	# parametri su cui il controllore può agire:
	# ξ → percentage of population vaccined per model step [0.0 - 0.03]
	# η → effect of the countermeasures [1.0 - 0.0) lower is better
	# θ → percentage of people under generalized lockdown [0.0 - 0.75] 1.0 is practically impossible.
	# θₜ → total number of days (model step) in which θ is applied
	# q → days of quarantine for each infected agent detected [0.0 - γ*2]
	# control_growth → rateo of growth [0.0 - 1.0] 0.0 means no growth, 1.0 means double
	# threshold_before_growth → threshold before increment controls. given by ncontrols / infected detected

	# population = 2500.0
	# df = model_params.read_data()
	# abm_parameters = model_params.extract_params(df, 8, population, 0.01)
	# model = graph.init(; abm_parameters...)
	# max_iter = 100
	# for i in 1:max_iter
	# 	println("[$i]/[$max_iter]")
	# 	data = graph.collect(model, graph.agent_step!, graph.model_step!; n=0)
	# 	show(data)
	# end

	function policy!(data::DataFrame, minimize, maximize; saveat=1, time_delay=90)
		# piglio i dati generati dall'ABM e li uso come base. 
		# so cosa voglio massimizzare e cosa minimizzare
		# uso un sistema di ODE come hybrid-model per le predizioni future
		# aggiorno i dati dell'ABM e vedo come procede.
		# nuovo screenshoot della situazione dopo time_delay passi (giorni)
		# se migliorato bene, se peggiorato sono cazzi 

		# https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
		# https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
		# https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
		# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
	end

	population = 2000.0
	df = model_params.read_data()
	abm_parameters = model_params.extract_params(df, 8, population, 0.01)
	model = graph.init(; abm_parameters...)
	data = graph.collect(model, graph.agent_step!, graph.model_step!; n=60)
	time_passed = [i for i in 1:length(data[!,1])]
	
	# To estimate the system, we first create a DataDrivenProblem, which requires measurement data. 
	# Since we want to use SINDy, we call solve with an sparsifying algorithm, in this case STLSQ 
	# which iterates different sparsity thresholds and returns a Pareto optimal solution. 
	# Note that we include the control signal in the basis as an additional variable c.
	prob = ContinuousDataDrivenProblem(Array(data)', time_passed, GaussianKernel())
	plot(prob)

	# Now we infer the system structure. First we define a Basis which collects all possible candidate terms. 
	@variables u[1:9] c[1:1]
	@parameters w[1:9]
	u = collect(u)
	c = collect(c)
	w = collect(w)

	h = Num[sin.(w[1] .* u[1]); cos.(w[2] .* u[1]); polynomial_basis(u, 5); c]

	basis = Basis(h, u, parameters=w, controls=c)

	using Random
	rng = Xoshiro(1234)
	sampler = DataProcessing(split = 0.8, shuffle = true, batchsize = 30, rng = rng)
	λs = exp10.(-10:0.1:0)
	opt = STLSQ(λs)
	res = solve(prob, basis, opt, options = DataDrivenCommonOptions(data_processing = sampler, digits = 1))
	
	# Where the resulting DataDrivenSolution stores information about the inferred model and the parameters:
	system = get_basis(res)
	params = get_parameter_map(system)

	plot(
		plot(prob), plot(res), layout = (1,2)
	)

	u,p,t = model_params.extract_params(df)
	prob = ODEProblem(uode.seir!, u, t, p)
	sol = solve(prob, Tsit5())
	plot(sol)
end
