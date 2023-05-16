module uode
	# implement a simple ODE system and solver for seir model
	# refers to. https://stackoverflow.com/questions/75902221/how-to-solve-the-error-undefvarerror-interpolatingadjoint-not-defined-using-d
	using OrdinaryDiffEq

	function seir!(du, u, p, t)
		S,E,I,R,D = u
		R₀, γ, σ, ω, δ = p
		dS = -R₀/γ*S + ω*R 
		dE = R₀/γ*S - σ*E 
		dI = σ*E - γ*I - δ*I
		dR = γ*I - ω*R 
		dD = δ*I
		du[1] = dS; du[2] = dE; du[3] = dI; du[4] = dR; du[5] = dD;
	end
end 