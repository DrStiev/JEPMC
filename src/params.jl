module model_params
    using CSV, Random, Distributions
    using DataFrames, DelimitedFiles
    using LinearAlgebra: diagind
	using Downloads, Measurements
	using Parameters
	using DrWatson: @dict

	# TODO: METTIMI A POSTO
	const population = 60_217_965

	# https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
	# https://covid19.who.int/WHO-COVID-19-global-data.csv
	# https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
	# dati per adatti a modello SIR
	function get_data(url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
		title = split(url,"/")
		df = DataFrame(CSV.File(
			Downloads.download(url, "data/"*title[length(title)]), 
			delim=",", header=1))
		return df
	end

	# TODO: make it general
	function extract_params(df; C=8, min_max_population=(50,5000), max_travel_rate=0.01, seed=1234)
		Random.seed!(seed)
		number_point_of_interest = rand(min_max_population[1]:min_max_population[2], C)
		migration_rate = zeros(C, C)
		for c in 1:C
			for c2 in 1:C
				migration_rate[c,c2] = (number_point_of_interest[c] + number_point_of_interest[c2]) / number_point_of_interest[c]
			end
		end
		maxM = maximum(migration_rate)
		migration_rate = (migration_rate .* max_travel_rate) ./ maxM
		migration_rate[diagind(migration_rate)] .= 1.0

		β = sum(df[!, :totale_positivi]) / population
		params =  @with_kw (T = length(df[!,1]), N = population,
			R₀_n = 1.6, R̅₀ = (t,p) -> p.R₀_n, γ = 1.0/18, σ = 1.0/5.2, 
			η = 1.0 / 20, δ₀ = sum(df[nrow(df), :deceduti]) / sum(df[!, :totale_positivi]),
			ω = 1.0/240, ψ = 0.03, ξ = 0.004, θ = 0.2, ϵ = 0.0,
			number_point_of_interest = number_point_of_interest, migration_rate = migration_rate);
		return params
	end
end
