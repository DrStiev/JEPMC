using Distributed # distributed computing

addprocs(4)
@time @everywhere include("graph_model.jl")

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
# parametri interattivi modello
params = Dict(
	:infection_period => 1:1:45,
	:detection_time => 1:1:21,
	:quarantine_time => 1:1:45,
)
fig, abmobs = interactive_graph_plot(model, agent_step!, model_step!, params)
abmobs
fig