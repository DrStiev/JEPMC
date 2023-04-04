# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/#Coupling-DifferentialEquations.jl-to-Agents.jl-1
# https://docs.sciml.ai/Overview/stable/getting_started/fit_simulation/#fit_simulation
# https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#first_sim
# https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
module ode_model
	using ModelingToolkit, DifferentialEquations, OrdinaryDiffEq
	using Plots, LaTeXStrings

	# TODO: add model diffusion
	function SEIRS!(du, u, p ,t)
		S, E, I, R = u
		N = sum(du)
		# β: rates of infection, γ: rates of recovery, σ: latency period 
		# ω: immunity period, μ: birth and background death, α: virus mortality
		β, γ, σ, ω, μ, α = p 
		# birth - infection + lost immunity - natural death
		du[1] = dS = μ*N - (β*I*S)/N + ω*R - μ*S
		# infection - latency - natural death
		du[2] = dE = (β*I*S)/N - σ*E - μ*E
		# latenza - recovery - natural + epidemic death
		du[3] = dI = σ*E - γ*I - (μ+α)*I
		# recovery - lost immunity - natual death
		du[4] = dR = γ*I - ω*R - μ*R
	end

	# u0 = [1.0-1E-3, 1E-3, 0.0, 0.0] # initial condition
	# tspan = (0.0, 1000.0) # ≈ 3 year
	# p = [3/14, 1/14, 1/7, 1/365, 1/365*76, 0.044]

	# prob = ODEProblem(seirs!, u0, tspan, p)
	# sol = solve(prob, Tsit5())

	# p1 = plot(sol, labels = [L"s" L"e" L"i" L"r"], title = "SEIRS Dynamics", lw = 2, xlabel = L"t")

	function get_ODE_problem(f, u0, tspan, p)
		return ODEProblem(f, u0, tspan, p)
	end

	function get_ODE_integrator(prob, method = Tsit5(); advance_to_tstop = true)
		return OrdinaryDiffEq.init(prob, method; advance_to_tstop = advance_to_tstop)
	end
end