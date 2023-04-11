module model_params
    using CSV, Random, Distributions
    using DataFrames, DelimitedFiles
    using LinearAlgebra: diagind
	using Downloads, Measurements
	using Parameters
	using DrWatson: @dict

	# dati per adatti a modello SIR
	function get_data(url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
		title = split(url,"/")
		df = DataFrame(CSV.File(
			Downloads.download(url, "data/"*title[length(title)]), 
			delim=",", header=1, select=[10, 11, 14]))
		df[!, :individui_sani] = [60_217_965 - i for i in df[!, 3]]
		return df
	end

	# TODO: extract real parameter
	function extract_params(df)
		s = df[1, 4]
		β = df[length(df[!,1]),1] / s
		params =  @with_kw (T = length(df[!,1]), N = 60_217_965,
			R₀_n = 1.6, R̅₀ = (t,p) -> p.R₀_n, γ = 1.0/18, σ = 1.0/5.2, 
			η = 1.0 / 20, δ₀ = df[length(df[!,2]),2] / df[length(df[!,1]),1],
			ω = 1.0/240, ψ = 0.03, ξ = 0.004, θ = 0.2);
			return params
	end
end
