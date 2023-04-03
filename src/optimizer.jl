# https://juliadynamics.github.io/Agents.jl/stable/examples/optim/
module optimizer
	using BlackBoxOptim, Random
	using Statistics: mean

	include("params.jl")
	include("graph.jl")

	# function cost(x)
	# 	graph_model.agent_step!
	# 	model = graph_model.model_init(;
	# 	Ns = x[1],
	# 	migration_rates = x[2],
	# 	death_rate = x[3],
	# 	)
end