module controller
	using DataFrames, DataDrivenDiffEq, DataDrivenSparse, LinearAlgebra, OrdinaryDiffEq, ModelingToolkit

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

	population = 2500.0
	df = model_params.read_data()
	abm_parameters = model_params.extract_params(df, 8, population, 0.01)
	model = graph.init(; abm_parameters...)
	max_iter = 100
	for i in 1:max_iter
		println("[$i]/[$max_iter]")
		data = graph.collect(model, graph.agent_step!, graph.model_step!; n=0)
		# aumento il numero di controlli sse ho una alta percentuale di infetti
		if model.properties[:infected_detected_ratio] ≥ 
			model.properties[:threshold_before_growth] # percentuale infetti
			model.properties[:ncontrols] *= model.properties[:control_growth]
		end
		show(data)
	end

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

	population = 2500.0
	df = model_params.read_data()
	abm_parameters = model_params.extract_params(df, 8, population, 0.01)
	model = graph.init(; abm_parameters...)
	data = graph.collect(model, graph.agent_step!, graph.model_step!; n=30)
	time_passed = length(data[!,1])
	# estrapolo andamento curve

end
