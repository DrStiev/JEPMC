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
		S = 1.0,
		E = 1E-3,
		I = 0.0,
		R = 0.0,
		tspan = (0.0, 1000.0),
		β = 3/14,
		γ = 1/14, 
		σ = 1/7,
		ω = 1/365, 
		μ = 1/365*76, 
		α = 0.044,
		)
		u0 = [S - E - I - R, E, I , R] # initial condition
		tspan = tspan # ≈ 3 year
		p = [β, γ, σ ,ω, μ, α]
		return u0, tspan, p
	end

    function dummyparams(;
        C = 8,
        max_travel_rate = 0.01,
        population_range = range(50,5000),
        infection_period = 14,
        reinfection_probability = 0.15,
        detection_time = 5,
        quarantine_time = 10,
        death_rate = 0.044,
        Is = [zeros(Int, C-1)..., 1],
        seed = 42,
        )

        Random.seed!(seed)
		Ns = rand(population_range, C)
		β_und = rand(0.2:0.1:0.8, C)
		β_det = β_und ./ 10
        exposure_time = rand(Exponential(2), C)

		Random.seed!(seed)
		migration_rates = zeros(C,C)
		for c in 1:C
			for c2 in 1:C
				migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
			end
		end
		maxM = maximum(migration_rates)
		migration_rates = (migration_rates .* max_travel_rate) ./ maxM
		migration_rates[diagind(migration_rates)] .= 1.0

		params = @dict(
			Ns,
			migration_rates,
			β_und,
			β_det,
			infection_period,
			detection_time,
			exposure_time,
			quarantine_time,
			reinfection_probability,
			death_rate,
			Is
		)
		return params
	end
end

# p = params.dummyparams(C = 8, max_travel_rate = 0.01)