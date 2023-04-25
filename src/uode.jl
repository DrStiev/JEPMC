module uode
	using OrdinaryDiffEq
	using ModelingToolkit
	using DataDrivenDiffEq
	using LinearAlgebra, DiffEqSensitivity, Optim
	using DiffEqFlux, Flux, Lux, Optimization, OptimizationOptimisers, OptimizationOptimJL
	using Plots, Random
	gr() # ← ?

	# https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/SEIR_exposure/seir_exposure.jl
	# https://www.youtube.com/watch?v=5zaB1B4hOnQ

	rng = Random.default_rng()
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
	
	function dudt_(u, p, t)
		S, E, I, R, N, D, C = u
		F, β0, α, κ, μ, σ, γ, d, λ = p # this was p_ 
		z = ann([S/N, I, D/N], p₀) 	# Exposure does not depend on exposed, removed, or cumulative!
		dS = -β0*S*F/N - z[1] -μ*S 	
		dE = β0*S*F/N + z[1] -(σ+μ)*E 
		dI = σ*E - (γ+μ)*I 
		dR = γ*I - μ*R 
		dN = -μ*N 
		dD = d*γ*I - λ*D 
		dC = σ*E 
	
		[dS, dE, dI, dR, dN, dD, dC]
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

	### Universal ODE Part 1
	# https://stackoverflow.com/questions/72535425/sciml-conflict-using-diffeqflux
	# Warning: FastChain is being deprecated in favor of Lux.jl. 
	# Lux.jl uses functions with explicit parameters f(u,p) like FastChain, 
	# but is fully featured and documented machine learning library.
	# ann = FastChain(FastDense(3, 64, tanh), FastDense(64, 64, tanh), FastDense(64, 1))
	# p = Float64.(initial_params(ann))
	ann = Lux.Chain(Lux.Dense(3 => 64, tanh), Lux.Dense(64 => 64, tanh), Lux.Dense(64 => 1))
	p₀, st = Lux.setup(rng, ann)
	const losses = []

	function get_uode(prob_nn, data)
		### use universal ode to compute first results
		# ERROR: Need an adjoint for constructor 
		# SciMLSensitivity.InterpolatingAdjoint{0, true, Val{:central}, 
		# SciMLSensitivity.ReverseDiffVJP{false}}. 
		# Gradient is of type ChainRulesCore.ZeroTangent
		function predict(θ) 
			Array(concrete_solve(prob_nn, Vern7(), u, θ, saveat=solution.t,
				abstol=1e-6, reltol=1e-6,
				sensealg=DiffEqFlux.InterpolatingAdjoint(autojacvec=DiffEqFlux.ReverseDiffVJP())))
		end
			
		# No regularisation right now
		function loss(θ)
			pred = predict(θ)
			sum(abs2, data[2:4,:] .- pred[2:4,:]), pred # + 1e-5*sum(sum.(abs, params(ann)))
		end
		
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
		# res1_uode = DiffEqFlux.sciml_train(loss, p₀, ADAM(0.01), cb=callback, maxiters = 500)
		# res2_uode = DiffEqFlux.sciml_train(loss, res1_uode.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
		adtype = Optimization.AutoZygote()
		optf = Optimization.OptimizationFunction((x, p₀) -> loss(x), adtype)
		# optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(p₀))
		optprob = Optimization.OptimizationProblem(optf, p₀)
		res_uode = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.01), callback=callback, maxiters=500)
		optprob2 = remake(optprob, u0=res_uode.u)
		res2_uode = Optimization.solve(optprob2, BFGS(initial_stepnorm=0.01), callback=callback, maxiters=10000, allow_f_increases=false)
		return res2_uode
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
		
		X = data 
		# Ideal derivatives
		DX = Array(solution(solution.t, Val{1})) 
		S, E, I, R, N, D, C = eachrow(X)
		F, β0, α, κ, μ, _, γ, d, λ = p 
		L = β.(0:tspan[end], β0, D, N, κ, α).*S.*I./N 
		L̂ = vec(ann([S./N I D./N]', res2_uode.minimizer))
		X̂ = [S./N I D./N]'
		
		# Create an optimizer for the SINDY problem
		opt = SR3()
		# Create the thresholds which should be used in the search process
		thresholds = exp10.(-6:0.1:1)
		return X̂, L̂, basis, thresholds, opt
	end
	
	function get_prediction(u, p, tspan1, tspan2)
		### create the data on which the model will be trained
		prob = ODEProblem(pandemic!, u, tspan1, p)
		solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=1)

		### actual solution from data
		prob = ODEProblem(pandemic!, u, tspan2, p)
		solution_extrapolate = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=1)

		# Ideal data
		tsdata = Array(solution)
		# noisy data
		data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))
		plot(abs.(tsdata - data))

		prob_nn = ODEProblem(dudt_, u, tspan1, p) # was p₀
		res2_uode = get_uode(prob_nn, data)
		loss(res2_uode.minimizer)
		plot(losses, yaxis=:log, xaxis=:log, xlabel="Iterations", ylabel="Loss")

		### Test on uode derivative data
		X̂, L̂, basis, thresholds, opt = get_SInDy_params(data, solution, p, tspan1, res2_uode)
		Ψ = SInDy(X̂[:, 2:end], L̂[2:end], basis, thresholds,  opt=opt, maxiter=10000, normalize=true, denoise=true) # Succeed
		println(Ψ.basis)

		### results
		a_prob = ODEProblem{false}(approx, u, tspan2, p)
		a_solution = solve(a_prob, Tsit5())
		p_uodesindy = scatter(solution_extrapolate, vars=[2,3,4], legend=:top_left, label=["True Exposed" "True Infected" "True Recovered"])
		plot!(p_uodesindy, a_solution, lw=5, vars[2,3,4], label=["Estimated Exposed" "Estimated Infected" "Estimated Recovered"])
		plot!(p_uodesindy, [20.99, 21.01], [0.0, maximum(hcat(Array(solution_extrapolate[2:4, :]), Array(a_solution[2:4,:])))], lw=5, color=:black, label="Training Data End")
		
		return a_solution
	end
end 