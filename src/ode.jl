module ode
	using OrdinaryDiffEq, StochasticDiffEq, Parameters, DiffEqGPU
	using LaTeXStrings, LinearAlgebra, Random, SparseArrays, Statistics
	using DataFrames, NaNMath

	# https://julia.quantecon.org/continuous_time/seir_model.html
	# https://julia.quantecon.org/continuous_time/covid_sde.html
	function F(u, p, t)
		S, E, I, R, D, R₀, δ = u 
		(;R̅₀, γ, σ, ψ, η, ξ, θ, δ₀, ω, ϵ) = p
		return [-γ*R₀*S*I + ω*R - ϵ*S;	# ds/dt
			γ*R₀*S*I - σ*E;				# de/dt
			σ*E - γ*I;					# di/dt
			(1-δ)*γ*I - ω*R + ϵ*S; 		# dr/dt
			δ*γ*I;						# dd/dt
			η*(R̅₀(t, p) - R₀);	 	 	 # dR₀/dt
			θ*(δ₀-δ);					# dδ/dt
		]
	end

	# without NaNMath -> domain error, due to probably the solver process
	# with NaNMath -> instability detected so aborting
	function G(u, p, t)
		S, E, I, R, D, R₀, δ = u 
		(;R̅₀, γ, σ, ψ, η, ξ, θ, δ₀, ω, ϵ) = p
		return [0; 0; 0; 0; 0; ψ*NaNMath.sqrt(R₀); ξ*NaNMath.sqrt(δ*(1-δ))]
	end

	get_SDE_problem(f, g, u0, tspan, p) = SDEProblem(f, g, u0, tspan, p)
	get_ODE_problem(f, u0, tspan, p) = ODEProblem(
		ODEFunction(f, syms = [:susceptible, :exposed, :infected, :recovered, :dead, :R₀, :mortality_rate]),
		u0, tspan, p)

	get_solution(prob::SciMLBase.SDEProblem, n = 100) = EnsembleSummary(
		solve(EnsembleProblem(prob), SOSRI(), EnsembleThreads(), trajectories = n))
	get_solution(prob) = DataFrame(solve(prob, Tsit5()))

	get_integrator(prob::SciMLBase.ODEProblem) = init(prob, Tsit5(); advance_to_tstop=true)
	get_integrator(prob::SciMLBase.SDEProblem) = init(prob, SOSRI(); advance_to_tstop=true)

	make_step!(integrator, step, stop_at_tdt) = step!(integrator, step, stop_at_tdt)
	notify_change_u!(integrator, is_change) = u_modified!(integrator, is_change)
end 
