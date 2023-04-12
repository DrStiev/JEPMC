module ode
	using OrdinaryDiffEq, StochasticDiffEq, Parameters
	using LaTeXStrings, LinearAlgebra, Random, SparseArrays, Statistics
	using NaNMath, DataFrames

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

	function G(u, p, t)
		S, E, I, R, D, R₀, δ = u 
		(;R̅₀, γ, ψ, η, ξ, θ, δ₀, ω) = p

		return [0; 0; 0; 0; ψ*NaNMath.sqrt(R₀); ξ*NaNMath.sqrt(δ*(1-δ))]
	end

	# get_SDE_problem(f, g, u0, tspan, p) = SDEProblem(f, g, u0, tspan, p)
	get_ODE_problem(f, u0, tspan, p) = ODEProblem(
		ODEFunction(f, syms = [:susceptible, :exposed, :infected, :recovered, :dead, :R₀, :mortality_rate]),
		u0, tspan, p)
	# function get_solution(prob::SciMLBase.SDEProblem, n = 100) 
	# 	ensembleprob = EnsembleProblem(prob)
	# 	# valutare se mettere EnsembleGPUArray()
	# 	sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), trajectories = 100)
	# 	return EnsembleSummary(sol)
	# end
	get_solution(prob) = DataFrame(solve(prob, Tsit5()))

	# https://julia.quantecon.org/continuous_time/seir_model.html
	# https://julia.quantecon.org/continuous_time/covid_sde.html
end 
