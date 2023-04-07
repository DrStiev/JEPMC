module model_params
    using CSV, Random, Distributions
    using DataFrames, DelimitedFiles
    using DrWatson: @dict
    using LinearAlgebra: diagind

    function extract_data_from_csv_ode(input)
        df = DataFrame(CSV.File(input, delim=",", header=1, select=[1, 5, 6, 8, 10]))
		df = df[df[!,:geoId] .== "IT", :]
		tspan = length(df[!, :popData2020]) # ≈ 3 year
		pop = maximum(df[!, :popData2020])
		S = pop
		E = 0
		I = df[tspan, 2]
		R = 0
		D = df[tspan, 3]
		d = sum(df[!, :deaths])
		i = sum(df[!, :cases])
		β = i / pop
		γ = 1/10.6
		σ = 1/5.5
		ω = 1/246.4
		α = d / i
		ϵ = 0
		ξ = 0
		u0 = [S - E - I - R- D, E, I , R, D] # initial condition
		p = [β, γ, σ ,ω, α, ϵ, ξ]
		return u0, tspan, p
    end

	function extract_data_from_csv_abm(input)
		df = DataFrame(CSV.File(input, delim=",", header=1, select=[1, 5, 6, 8, 10]))
		df = df[df[!,:geoId] .== "IT", :]
	end

	function ode_dummyparams(;
		S = 5.9641488E7, E = 0, I = 229,
		R = 0, D = 0, 
		tspan = (0.0, 1000.0),
		β = 3.9166E-1, # rates of infection
		γ = 1/14, # rates of recover
		σ = 1/4, # latency period 
		ω = 1/240, # immunity period
		α = 7.64E-3, # virus mortality
		ϵ = 0.0, # vaccine rateo
		ξ = 0.0, # strong immune system
		)
		
		u0 = [S, E, I , R, D] # initial condition
		tspan = tspan # ≈ 3 year
		p = [β, γ, σ ,ω, α, ϵ, ξ]
		return u0, tspan, p
	end

    function g_dummyparams(;
        C = 8,
        max_travel_rate = 0.01,
        population_range = range(50,5000),
        β = 6/14, γ = 1/14, σ = 1/4, ω = 1/240, 
		α = 9E-3, ϵ = 0.0, ξ = 0.0, seed = 42,
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
			β, γ, σ ,ω, α, ϵ, ξ,
		)
		return params
	end

	function c_dummyparams(;
		N = 1E3, initial_infected = 1,
		β = 6/14, γ = 1/14, σ = 1/4, ω = 1/240, 
		α = 9E-3, ϵ = 0.0, ξ = 0.0,
        interaction_radius = 0.012,
        dt = 1.0, speed = 0.002,
        space_dimension = (1.0, 1.0),
        spacing = 0.02, steps_per_day = 24,
		)
		return @dict(
			N, initial_infected,
			β, γ, σ ,ω, α, ϵ, ξ,
			interaction_radius,
			dt, speed, space_dimension,
			spacing, steps_per_day,
		)
	end
end