module model_params
    using CSV, Random, Distributions
    using DataFrames, DelimitedFiles
    using DrWatson: @dict
    using LinearAlgebra: diagind

    function extract_param_from_csv(input)
        df = DataFrame(CSV.File(input))
        params = Dict(pairs(eachcol(df)))
        return params
    end

	function ode_dummyparams(;
		S = 2.2087E4, E = 0.0, I = 1.0,
		R = 0.0, D = 0.0, 
		tspan = (0.0, 1000.0),
		β = 6/14, # rates of infection
		γ = 1/20, # rates of recover
		σ = 1/5.8, # latency period 
		ω = 1/240, # immunity period
		α = 9E-3, # virus mortality
		ξ = 8.4E-3, # vaccine per day
		)
		
		u0 = [S, E, I , R, D] # initial condition
		tspan = tspan # ≈ 3 year
		p = [β, γ, σ ,ω, α, ξ]
		return u0, tspan, p
	end

    function dummyparams(;
        C = 8,
        max_travel_rate = 0.01,
        population_range = range(50,5000),
        β = 6/14, γ = 1/20, σ = 1/5.8, ω = 1/240, 
		α = 9E-3, ξ = 8.4E-3, seed = 42,
    	)

        Random.seed!(seed)
		number_point_of_interest = rand(population_range, C)
		migration_rate = zeros(C,C)
		for c in 1:C
			for c2 in 1:C
				migration_rate[c, c2] = (number_point_of_interest[c] + number_point_of_interest[c2]) / number_point_of_interest[c]
			end
		end
		maxM = maximum(migration_rate)
		migration_rate = (migration_rate .* max_travel_rate) ./ maxM
		migration_rate[diagind(migration_rate)] .= 1.0

		params = @dict(
			number_point_of_interest,
			migration_rate,
			β, γ, σ, ω, α, ξ,
		)
		return params
	end

	function c_dummyparams(;
		infection_period = 14,
        detection_time = 5,
        exposure_time = 5,
        immunity_period = 270, 
        interaction_radius = 0.012,
        dt = 1.0,
        speed = 0.002,
        death_rate = 0.009, 
        N = 22087,
        initial_infected = 1,
        βmin = 0.2,
        βmax = 0.8,
        space_dimension = (1.0, 1.0),
        spacing = 0.02,
		steps_per_day = 24,
		)
		params = @dict(
			infection_period,
			detection_time,
			exposure_time,
			immunity_period, 
			interaction_radius,
			dt,
			speed,
			death_rate, 
			N,
			initial_infected,
			βmin,
			βmax,
			space_dimension,
			spacing,
			steps_per_day,
		)
		return params
	end
end