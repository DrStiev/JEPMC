# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/#Coupling-DifferentialEquations.jl-to-Agents.jl-1
# https://docs.sciml.ai/Overview/stable/getting_started/fit_simulation/#fit_simulation
# https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#first_sim
# https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
module ode
	using ModelingToolkit, DifferentialEquations, OrdinaryDiffEq
	using Plots, LaTeXStrings

	# model SEIQR(V)D con perdita di immunità
	function SEIQRD!(du, u, p ,t)
		S, E, I, Q, R, D = u
		# popolazione non conta i decessi
		N = sum(u) - D
		# β: rates of infection
		# γ: rates of recovery, 
		# σ: latency period 
		# ω: immunity period, 
		# α: virus mortality, 
		# δ: probability of quarantine
		# ξ: rate of vaccination
		β, γ, σ, ω, α, δ, ξ = p 
		# total - infection + lost immunity - vaccination
		du[1] = dS = N - (β*I*S)/N + ω*R - ξ*S
		# infection - latency
		du[2] = dE = (β*I*S)/N - σ*E
		# latenza - recovery - epidemic death - quarantine
		du[3] = dI = σ*E - γ*I - α*I - δ*Q
		# quarantena - recovery - epidemic death
		du[4] = dQ = δ*I - γ*Q - α*Q
		# recovery - lost immunity + vaccine + recovery quarantena
		du[5] = dR = γ*I - ω*R - μ*R + ξ*S + γ*Q
		# epidemic death infect + epidemic death quarantined
		du[6] = dD = α*I + α*Q
	end

	function get_ODE_problem(f, u0, tspan, p)
		return ODEProblem(f, u0, tspan, p)
	end

	function get_ODE_integrator(prob, method = Tsit5(); advance_to_tstop = true)
		return OrdinaryDiffEq.init(prob, method; advance_to_tstop = advance_to_tstop)
	end

	function get_solution(prob)
		return solve(prob)
	end

	function line_plot(sol, labels = [L"Susceptible" L"Exposed" L"Infected" L"Quarantine" L"Recovered" L"Dead"], title = "SEIRS Dynamics")
		my_range = LinRange(0, sum(sol[1]), 11)
		p = plot(sol, labels = labels, title = title, 
			lw = 2, yticks = (my_range, 0:0.1:1), xlabel = L"Days")
		return p
	end
end 
