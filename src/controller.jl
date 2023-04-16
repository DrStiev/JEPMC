module controller
    using ModelingToolkit
	using DataDrivenDiffEq
	using LinearAlgebra, DiffEqSensitivity, Optim
	using DiffEqFlux, Flux
	using Plots

    include("ode.jl")

    # https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/SEIR_exposure/seir_exposure.jl
	# https://www.youtube.com/watch?v=5zaB1B4hOnQ
end
