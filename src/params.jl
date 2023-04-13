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
	function get_data(url="https://covid19.who.int/WHO-COVID-19-global-data.csv")
		title = split(url,"/")
		df = DataFrame(CSV.File(
			Downloads.download(url, "data/"*title[length(title)]), 
			delim=",", header=1))
		# df = df[df.Country_code .== "IT", :]
		# select!(df, [:New_cases, :New_deaths])
		# for i in 1:ncol(df)
		# 	df[!, i] = df[!, i] ./ population
		# end
		return df
	end

	# TODO: make it general
	function extract_params(df)
		β = sum(df[!, :New_cases]) / population
		params =  @with_kw (T = length(df[!,1]), N = population,
			R₀_n = 1.6, R̅₀ = (t,p) -> p.R₀_n, γ = 1.0/18, σ = 1.0/5.2, 
			η = 1.0 / 20, δ₀ = sum(df[!, :New_deaths]) / sum(df[!, :New_cases]),
			ω = 1.0/240, ψ = 0.03, ξ = 0.004, θ = 0.2, ϵ = 0.0);
		return params
	end
end
