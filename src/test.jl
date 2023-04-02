using Pkg
Pkg.activate(".")
Pkg.instantiate()
# Pkg.precompile()
# Pkg.resolve()

@btime include("file_reader.jl")
@btime include("graph_model.jl")
@btime include("graph_plot.jl")
@btime include("controller.jl")

@btime params = file_reader.extract_param_from_csv("csv_files/example.csv")

params = graph_model.create_params(
	C = 8,
	min_population = 50,
	max_population = 5000,
	max_travel_rate = 0.01, 
	infection_period = 18, 
	reinfection_probability = 0.05,
	detection_time = 5,
	exposure_time = 5,
	quarantine_time = 14,
	death_rate = 0.044,
	)
@btime model = graph_model.model_init(; params...)
@btime fig, data = graph_plot.line_plot(model, graph_model.agent_step!, 100)
fig