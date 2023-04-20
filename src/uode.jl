module uode
	using OrdinaryDiffEq
	using ModelingToolkit
	using DataDrivenDiffEq
	using LinearAlgebra, DiffEqSensitivity, Optim
	using DiffEqFlux, Flux, Lux, Optimization
	using Plots

	# https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/SEIR_exposure/seir_exposure.jl
	# https://www.youtube.com/watch?v=5zaB1B4hOnQ

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
	
	getODEProblem(F, u0, tspan, p) = ODEProblem(F, u0, tspan, p)
	getODEProblem2(F, u0, tspan, p) = ODEProblem{false}(F, u0, tspan, p)
	getVern7Solution(prob; abstol=1e-12, reltol=1e-12, saveat=1) = solve(prob, Vern7(), abstol=abstol, reltol=reltol, saveat=saveat)
	getTsit5Solution(prob) = solve(prob, Tsit5())
	getConcreteSolverTsit5(prob, u0, p; saveat=1) = concrete_solve(prob, Tsit5(), u0, p, saveat = saveat)
	getConcreteSolverVern7(prob, u0, θ; abstol=1e-6, reltol=1e-6, saveat=1) = 
		concrete_solve(prob, Vern7(), u0, θ, saveat=saveat,
			abstol=abstol, reltol=reltol, sensealg=DiffEqFlux.InterpolatingAdjoint(autojacvec=DiffEqFlux.ReverseDiffVJP()))

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
	
	### Universal ODE Part 2: SInDy to Equations
	
	function get_SInDy_params(data, solution, p, tspan, res2_uode)
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
		DX = Array(solution(solution.t, Val{1})) # solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)
		S, E, I, R, N, D, C = eachrow(X)
		F, β0, α, κ, μ, _, γ, d, λ = p # p_ # [10.0, 0.5944, 0.4239, 1117.3, 0.02, 1/3, 1/5,0.2, 1/11.2]
		L = β.(0:tspan[end], β0, D, N, κ, α).*S.*I./N # tspan = (0.0, 21.0)
		L̂ = vec(ann([S./N I D./N]', res2_uode.minimizer))
		X̂ = [S./N I D./N]'
		
		# Create an optimizer for the SINDY problem
		opt = SR3()
		# Create the thresholds which should be used in the search process
		thresholds = exp10.(-6:0.1:1)
		return X̂, L̂, basis, thresholds, opt
	end
	
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
	
	function get_prediction(u, p, t1, t2)
		### create the data on which the model will be trained
		R0 = p[2]/p[7]*p[6]/(p[6]+p[5])
		prob = getODEProblem(pandemic!, u, t1, p)
		solution = getVern7Solution(prob)

		### actual solution from data
		prob = getODEProblem(pandemic!, u, t2, p)
		solution_extrapolate = getVern7Solution(prob)

		# Ideal data
		tsdata = Array(solution)
		# noisy data
		data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))

		### use universal ode to compute first results
		predict(θ) = Array(getConcreteSolverVern7(prob_nn, u0, θ))
		
		# No regularisation right now
		function loss(θ)
			pred = predict(θ)
			sum(abs2, data[2:4,:] .- pred[2:4,:]), pred # + 1e-5*sum(sum.(abs, params(ann)))
		end
		
		# unsupported `const` declaration on local variable
		# const losses = []
		losses = []
		callback(θ, l, pred) = begin
			push!(losses, l)
			if length(losses) % 50 == 0
				println(losses[end])
			end
			false
		end

		prob_nn = getODEProblem(dudt_, u, t1, p)
		s = getConcreteSolverTsit5(prob, u, p)

		# Warning: sciml_train is being deprecated in favor of direct usage of Optimization.jl. 
		# Please consult the Optimization.jl documentation for more details. 
		# Optimization.jl's PolyOpt solver is the polyalgorithm of sciml_train
		res1_uode = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 500)
		res2_uode = DiffEqFlux.sciml_train(loss, res1_uode.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
	
		### Test on uode derivative data
		X̂, L̂, basis, thresholds, opt = get_SInDy_params(data, solution, p, t1, res2_uode)
		Ψ = SInDy(X̂[:, 2:end], L̂[2:end], basis, thresholds,  opt=opt, maxiter=10000, normalize=true, denoise=true) # Succeed
		# println(Ψ.basis)

		### results
		a_prob = getODEProblem2(approx, u, t2, p)
		a_solution = getTsit5Solution(a_prob)
		return a_solution
	end
end 
