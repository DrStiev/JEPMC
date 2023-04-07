module ode
	using ModelingToolkit, DifferentialEquations, OrdinaryDiffEq
	using Plots, LaTeXStrings

	# model SEIRD con perdita di immunità
	function SEIRD!(du, u, p, t)
		S, E, I, R, D = u
		N = sum(u)
		# β: rates of infection :S → :E
		# γ: rates of recover, :I → :R
		# σ: latency period, :E → :I 
		# ω: immunity period, :R → :S
		# α: virus mortality, :I → :D
		# ϵ: vaccine rateo, :S → :R
		# ξ: strong immune system, :E → :S
		β, γ, σ, ω, α, ϵ, ξ = p 
		du[1] = dS = -β*I*S/N + ω*R - ϵ*S + ξ*E 
		du[2] = dE = β*I*S/N - σ*E - ξ*E 
		du[3] = dI = σ*E - γ*I - α*I 
		du[4] = dR = γ*I - ω*R + ϵ*S
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

	function area_plot(sol, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered" L"Dead"], title = "SEIRD Dynamics")
		p = areaplot(sol.t, sol', labels = labels, title = title, xlabel = L"Days")
		return p
	end
end 
