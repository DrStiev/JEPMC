module controller
	using DataFrames, DataDrivenDiffEq, DataDrivenSparse
	using LinearAlgebra, OrdinaryDiffEq, ModelingToolkit
	using Plots, Random
	include("graph.jl")
	include("params.jl")
	include("uode.jl")

	# parametri su cui il controllore può agire:
	# ξ → percentage of population vaccined per model step [0.0 - 0.03]
	# η → countermeasures [1.0 - 0.0) lower is better
	# θ → percentage of people under generalized lockdown [0.0 - 1.0).
	# θₜ → total number of days (model step) in which θ is applied
	# q → days of quarantine for each infected agent detected [0.0 - γ*2]

	function policy!(data::DataFrame, minimize, maximize; saveat=1, time_delay=90)
		# https://docs.sciml.ai/SciMLSensitivity/dev/getting_started/
		# https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/parameter_estimation_ode/#odeparamestim
		# https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
		# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
	end
end
