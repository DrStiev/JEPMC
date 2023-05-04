module controller
	#include("uode.jl")
	

    # https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/SEIR_exposure/seir_exposure.jl
	# https://www.youtube.com/watch?v=5zaB1B4hOnQ

	# mando in run il modello per un tot numero di step, esempio 21.
	# prendo le informazioni del model.collect e le trasformo in matrice 
	# con Array(df) oppure Matrix(df). 
	# provo a predire la curva di infetti e di happiness senza alcun intervento
	# cerco di minimizzare la curva infetti e massimizzare la curca happiness.
	# in questo modo dovrei avere un array di parametri che mi definisce 
	# quali sono i migliori parametri da utilizzare. li applico e vedo che succedde.

	# parametri su cui il controllore può agire:
	# ξ → percentage of population vaccined per model step [0.0 - 0.03]
	# η → effect of the countermeasures [1.0 - 0.0) lower is better
	# θ → percentage of people under generalized lockdown [0.0 - 0.75] 1.0 is practically impossible.
	# θₜ → total number of days (model step) in which θ is applied
	# q → days of quarantine for each infected agent detected [0.0 - γ*2]
	# control_growth → rateo of growth [0.0 - 1.0] 0.0 means no growth, 1.0 means double
	# threshold_before_growth → threshold before increment controls. given by ncontrols / infected detected

	include("graph.jl")
	include("params.jl")
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
		display(data)
	end
	model.ncontrols
	function get_data(field_to_maximize, field_to_minimize;time_interval=21)

	end

end
