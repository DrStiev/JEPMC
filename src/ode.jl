module ode
	using ModelingToolkit, DifferentialEquations, OrdinaryDiffEq
	using Plots, LaTeXStrings

	# model SEIR(V)D con perdita di immunità
	function SEIRD!(du, u, p, t)
		S, E, I, R, D = u
		N = sum(u)
		# β: rates of infection
		# γ: rates of recover, 
		# σ: latency period 
		# ω: immunity period, 
		# α: virus mortality, 
		# ξ: rate of vaccination
		β, γ, σ, ω, α, ξ = p 
		du[1] = dS = -β*I*S/N + ω*R - ξ*S 
		du[2] = dE = β*I*S/N - σ*E
		du[3] = dI = σ*E - γ*I - α*I
		du[4] = dR = γ*I - ω*R + ξ*S
		du[5] = dD = α*I
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

	function line_plot(sol, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered" L"Dead"], title = "SEIRD Dynamics")
		p = plot(sol, labels = labels, title = title, 
			lw = 2, xlabel = L"Days")
		return p
	end
end 
