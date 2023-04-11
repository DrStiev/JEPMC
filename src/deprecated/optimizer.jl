# https://docs.sciml.ai/Overview/stable/getting_started/fit_simulation/#fit_simulation
# TODO: test me
module optimizer
	using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
	using ForwardDiff, Plots

	function loss(prob, newp)
		newprob = remake(prob, p = newp)
		sol = solve(newprob, saveat = 1)
		loss = sum(abs2, sol .- data)
		return loss, sol
	end

	callback = function(p, l, sol)
		display(l)
		plt = plot(sol, label = "Current Prediction")
		scatter!(plt, datasol, label = "Data")
		display(plt)
		return false
	end

	function opt(prob, pguess)
		adtype = Optimization.AutoForwardDiff()
		optf = Optimization.OptimizationFunction((x, y, p) -> loss(x, y), adtype)
		optprob = Optimization.OptimizationProblem(optf, pguess)
		return Optimization.solve(optprob, PolyOpt(), callback = callback, maxiters = 200)
	end
end

