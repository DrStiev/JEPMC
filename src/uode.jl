module uode
	using OrdinaryDiffEq
	using ModelingToolkit
	using DataDrivenDiffEq
	using LinearAlgebra, DiffEqSensitivity, Optim
	using DiffEqFlux, Flux, Lux, Optimization

	# https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/SEIR_exposure/seir_exposure.jl
	# https://www.youtube.com/watch?v=5zaB1B4hOnQ

	# TODO: implement time-dependant parameters? 
	# https://stackoverflow.com/questions/52311652/time-dependent-events-in-ode
	# https://discourse.julialang.org/t/time-dependent-events-in-ode/14951/3
	# https://discourse.julialang.org/t/differentialequations-jl-solving-ode-with-time-dependent-parameter-allocation-friendly/78284

	# seir equations to compute a general pandemic model
	function pandemic!(du, u, p, t)
		S, E, I, R, N, D, C = u
		F, β0, α, κ, μ, σ, γ, d, λ = p

		dS = -β0*S*F/N - β(t, β0, D, N, κ, α)*S*I/N -μ*S 		# susceptible
		dE = β0*S*F/N + β(t, β0, D, N, κ, α)*S*I/N -(σ+μ)*E 	# exposed
		dI = σ*E - (γ+μ)*I 										# infected
		dR = γ*I - μ*R 											# removed (recovered + dead)
		dN = -μ*N 												# total population
		dD = d*γ*I - λ*D 										# severe, critical cases, and deaths
		dC = σ*E 												# +cumulative cases
	
		du[1] = dS; du[2] = dE; du[3] = dI; du[4] = dR
		du[5] = dN; du[6] = dD; du[7] = dC
	end

	# compute exposure
	β(t, β0, D, N, κ, α) = β0*(1-α)*(1-D/N)^κ
	
	getODEProblem(F, u0, tspan, p) = ODEProblem(pandemic!, u0, tspan, p_)
	getSolution(prob) = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)
	getConcreteSolver(prob, saveat) = concrete_solve(prob, Tsit5(), u0, p, saveat = saveat)
	
	### Universal ODE Part 1
	# https://stackoverflow.com/questions/72535425/sciml-conflict-using-diffeqflux
	# Warning: FastChain is being deprecated in favor of Lux.jl. 
	# Lux.jl uses functions with explicit parameters f(u,p) like FastChain, but is fully featured and documented machine learning library.
	ann = FastChain(FastDense(3, 64, tanh), FastDense(64, 64, tanh), FastDense(64, 1))
	p = Float64.(initial_params(ann))
	
	function dudt_(u, p, t)
		S, E, I, R, N, D, C = u
		F, β0, α, κ, μ, σ, γ, d, λ = p
		z = ann([S/N, I, D/N], p) 	# Exposure does not depend on exposed, removed, or cumulative!
		dS = -β0*S*F/N - z[1] -μ*S 	
		dE = β0*S*F/N + z[1] -(σ+μ)*E 
		dI = σ*E - (γ+μ)*I 
		dR = γ*I - μ*R 
		dN = -μ*N 
		dD = d*γ*I - λ*D 
		dC = σ*E 
	
		[dS, dE, dI, dR, dN, dD, dC]
	end

	# prob_nn = ODEProblem(dudt_, u0, tspan, p)
	# s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat=1)
	
	function predict(θ)
		Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = solution.t,
							 abstol = 1e-6, reltol = 1e-6,
							 sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
	end
	
	# No regularisation right now
	function loss(θ)
		pred = predict(θ)
		sum(abs2, data[2:4,:] .- pred[2:4,:]), pred # + 1e-5*sum(sum.(abs, params(ann)))
	end
	
	# loss(p) # ← ?
	
	const losses = []
	callback(θ, l, pred) = begin
		push!(losses, l)
		if length(losses) % 50 == 0
			println(losses[end])
		end
		false
	end
	
	# Warning: sciml_train is being deprecated in favor of direct usage of Optimization.jl. 
	# Please consult the Optimization.jl documentation for more details. 
	# Optimization.jl's PolyOpt solver is the polyalgorithm of sciml_train
	res1_uode = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 500)
	res2_uode = DiffEqFlux.sciml_train(loss, res1_uode.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

	# loss(res2_uode.minimizer)
	
	### Universal ODE Part 2: SInDy to Equations
	
	# Create a Basis
	@variables u[1:3]
	# Lots of polynomials
	polys = Operation[]
	for i ∈ 0:2, j ∈ 0:2, k ∈ 0:2
		push!(polys, u[1]^i * u[2]^j * u[3]^k)
	end
	
	# And some other stuff
	h = [cos.(u)...; sin.(u)...; unique(polys)...]  # ← ?
	basis = Basis(h, u)
	
	X = data # Array(solution) + Float32(1e-5)*randn(eltype(Array(solution)), size(Array(solution)))
	# Ideal derivatives
	DX = Array(solution(solution.t, Val{1})) # exact solution from getSolution
	S, E, I, R, N, D, C = eachrow(X)
	F, β0, α, κ, μ, _, γ, d, λ = p_
	L = β.(0:tspan[end], β0, D, N, κ, α).*S.*I./N # tspan = (0.0, 21.0)
	L̂ = vec(ann([S./N I D./N]', res2_uode.minimizer))
	X̂ = [S./N I D./N]'
	
	# Create an optimizer for the SINDY problem
	opt = SR3()
	# Create the thresholds which should be used in the search process
	thresholds = exp10.(-6:0.1:1)
	
	# Test on uode derivative data
	Ψ = SInDy(X̂[:, 2:end], L̂[2:end], basis, thresholds,  opt = opt, maxiter = 10000, normalize = true, denoise = true) # Succeed
	# println(Ψ.basis)
	
	# Build a ODE for the estimated system
	function approx(u, p, t)
		S, E, I, R, N, D, C = u
		F, β0, α, κ, μ, σ, γ, d, λ = p
		z = Ψ([S/N, I, D/N]) 			# Exposure does not depend on exposed, removed, or cumulative!
		
		dS = -β0*S*F/N - z[1] -μ*S 		# susceptible
		dE = β0*S*F/N + z[1] -(σ+μ)*E 	# exposed
		dI = σ*E - (γ+μ)*I 				# infected
		dR = γ*I - μ*R 					# removed (recovered + dead)
		dN = -μ*N 						# total population
		dD = d*γ*I - λ*D 				# severe, critical cases, and deaths
		dC = σ*E 						# +cumulative cases
	
		[dS, dE, dI, dR, dN, dD, dC]
	end
	
	# Create the approximated problem and solution
	a_prob = ODEProblem{false}(approx, u0, tspan2, p_)
	a_solution = solve(a_prob, Tsit5())
	
	# p_uodesindy = scatter(solution_extrapolate, vars=[2,3,4], legend = :topleft, label=["True Exposed" "True Infected" "True Recovered"])
	# plot!(p_uodesindy,a_solution, lw = 5, vars=[2,3,4], label=["Estimated Exposed" "Estimated Infected" "Estimated Recovered"])
	# plot!(p_uodesindy,[20.99,21.01],[0.0,maximum(hcat(Array(solution_extrapolate[2:4,:]),Array(_sol_uode[2:4,:])))],lw=5,color=:black,label="Training Data End")
	
	# savefig("universalodesindy_extrapolation.png")
	# savefig("universalodesindy_extrapolation.pdf")
end 
