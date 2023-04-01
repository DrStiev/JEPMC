using Pkg
Pkg.activate(".")
Pkg.instantiate()
# Pkg.precompile()
# Pkg.resolve()

@time include("file_reader.jl")
@time include("graph_model.jl")
@time include("graph_plot.jl")
@time include("controller.jl")

params = graph_model.create_params(
	C = 8,
	min_population = 50,
	max_population = 5000,
	max_travel_rate = 0.01, 
	infection_period = 18, 
	reinfection_probability = 0.15,
	detection_time = 5,
	quarantine_time = 14,
	death_rate = 0.044,
	)
@time model = graph_model.model_init(; params...)
# @time graph_plot.hist_animation(model, 100)
@time graph_plot.line_plot(model, 200)

abmobs = graph_model.get_observable(model; graph_model.agent_step!)