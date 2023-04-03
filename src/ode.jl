# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/#Coupling-DifferentialEquations.jl-to-Agents.jl-1
# https://docs.sciml.ai/Overview/stable/getting_started/fit_simulation/#fit_simulation
# https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#first_sim
# https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
module ode_model
	using ModelingToolkit, DifferentialEquations, OrdinaryDiffEq
	using Plots, LaTeXStrings

	function seirs!(du, u, p ,t)
		S, E, I, R = u
		N = S + E + I + R
		β, γ, σ, ξ, δ = p # inf, inf_per, exp_per, reinf_prob, death_rate
		du[1] = dS = - (β * S * I) / N + ξ * R 
		du[2] = dE = (β * S * I) / N - σ * E 
		du[3] = dI = σ * E - γ * I 
		du[4] = dR = γ * I - ξ * R
	end

	# u0 = [1.0-1E-7-4.0*1E-7, 4.0*1E-7, 1E-7, 0.0] # initial condition
	# tspan = (0.0, 365.0) # ≈ 1 year
	# p = [1/1.65, 1/18, 1/5.2, 1/6.67, 1/22.72]

	# prob = ODEProblem(seirs!, u0, tspan, p)
	# sol = solve(prob, Tsit5())

	# p1 = plot(sol, labels = [L"s" L"e" L"i" L"r" L"d"], title = "SEIRS Dynamics", lw = 2, xlabel = L"t")

	function get_ODE_problem(f, u0, tspan, p)
		return ODEProblem(f, u0, tspan, p)
	end

	function get_ODE_integrator(prob, method = Tsit5(); advance_to_tstop = true)
		return OrdinaryDiffEq.init(prob, method; advance_to_tstop = advance_to_tstop)
	end
end