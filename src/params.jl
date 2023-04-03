module file_reader
    using CSV, Random, Distributions
    using DataFrames, DelimitedFiles
    using DrWatson: @dict
    using LinearAlgebra: diagind

    include("optimizer.jl")

    function extract_param_from_csv(input)
        df = DataFrame(CSV.File(input))
        params = Dict(pairs(eachcol(df)))
        return params
    end

    function dummyparams(;
        C,
        max_travel_rate,
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
		β_und = rand(0.2:0.1:0.7, C)
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

p = file_reader.dummyparams(C = 8, max_travel_rate = 0.01)