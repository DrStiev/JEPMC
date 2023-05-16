module uode
	# implement a simple ODE system and solver for seir model
	# refers to. https://stackoverflow.com/questions/75902221/how-to-solve-the-error-undefvarerror-interpolatingadjoint-not-defined-using-d
	using OrdinaryDiffEq

	function seir!(du, u, p, t)
		S,E,I,R,D = u
		R₀, γ, σ, ω, δ = p
		dS = -R₀/γ*S + 1/ω*R 
		dE = R₀/γ*S - 1/σ*E 
		dI = 1/σ*E - 1/γ*I - δ*I
		dR = 1/γ*I - 1/ω*R 
		dD = δ*I
		du[1] = dS; du[2] = dE; du[3] = dI; du[4] = dR; du[5] = dD;
	end

	function get_ode_problem(F, u, tspan, p)
		return ODEProblem(F, u, tspan, p)
	end

	function get_ode_solution(prob)
		return solve(prob, Tsit5())
	end
end 