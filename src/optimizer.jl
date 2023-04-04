# https://juliadynamics.github.io/Agents.jl/stable/examples/optim/
module optimizer
# FIXME: not finished!
	using BlackBoxOptim, Random
	using Statistics: mean
	using Agents

	@time include("graph.jl")
	@time include("params.jl")

	function cost(x)
		model = graph_model.model_init(; x...)

		infected_fraction(model) = 
			count(a.status ==:I for a in allagents(model))/nagents(model)
		quarantined_fraction(model) = 
			count(a.status ==:Q for a in allagents(model))/nagents(model)

		_, mdf = run!(
			model, 
			graph.agent_step!,
			50;
			mdata = [infected_fraction, quarantined_fraction],
			when_model = [50],
			# replicates = 10, # param not exists
		)

		return mdf.infected_fraction
	end

	Random.seed!(10)
	x0 = model_params.dummyparams()
	m = mean(cost(x0) for _ in 1:10)
	# c = cost(x0)
end