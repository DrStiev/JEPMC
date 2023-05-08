module uode
	# implement a simple ODE system and solver for seir model
	# refers to. https://stackoverflow.com/questions/75902221/how-to-solve-the-error-undefvarerror-interpolatingadjoint-not-defined-using-d
	using OrdinaryDiffEq

	function seir!(du, u, p, t)
		S,E,I,R,D = u
		R₀, γ, σ, ω, ξ, δ, η, ϵ, q, θ, θₜ = p
		β = R₀*γ*η
		dS = -β*S + ω*R + ϵ*E - ξ*S
		dE = β*S - σ*E - ϵ*E
		dI = σ*E - γ*I - δ*I
		dR = γ*I - ω*R + ξ*S
		dD = δ*I
		du[1] = dS; du[2] = dE; du[3] = dI; du[4] = dR; du[5] = dD;
	end
end 