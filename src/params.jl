module model_params
    using CSV, Random, Distributions, DataFrames, DelimitedFiles
	using Statistics: mean
    using LinearAlgebra: diagind
	using Downloads, Parameters, DifferentialEquations, LsqFit
	using DrWatson: @dict

	# FIXME: not very flexible
	const population = 60_217_965

	# https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
	# https://covid19.who.int/WHO-COVID-19-global-data.csv
	# https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
	function get_data(path, url="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
		title = split(url,"/")
		isdir(path) == false && mkpath(path)
		df = DataFrame(CSV.File(
			Downloads.download(url, path*title[length(title)]), 
			delim=",", header=1))
		return df
	end

	# https://nextjournal.com/berwa/sir-model-for-covid-19-estimating-R_o
	function estimate_R₀(df, β, γ, S₀, I₀, R₀, tspan)
		function sir_ode!(du, u, p, t)
			S, I, R = u
			β, γ = p

			dS = -β*S*I
			dI = +β*S*I - γ*I
			dR = + γ*I

			du .= (dS, dI, dR)
			du
		end
		prob = ODEProblem(sir_ode!, [S₀, I₀, R₀], tspan, [β, γ])
		sol = solve(prob, saveat=0.1);
		model(R, param) = population .-R -population .* exp.((-param[1]/population).*R)
		params = []
		active_infections_vec = df[!, :totale_casi] - df[!, :dimessi_guariti]
		for i in 1:10
			p₀ = [randn()]
			global fit = curve_fit(model, df[!, :dimessi_guariti], active_infections_vec, p₀)
			push!(params, fit.param[1])
		end
		return confidence_interval(fit, 0.05)
	end

	function extract_params(df, C, min_max_population, max_travel_rate, seed=1234)
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

		T = length(df[!,1])

		i = df[1,:nuovi_positivi]/population
		r = df[1,:dimessi_guariti]/population

		γ = 1.0/14 
		σ = 1.0/5.6 
		ω = 1.0/240
		ξ = 0.0
		δ = df[nrow(df), :deceduti] / sum(df[!, :nuovi_positivi])
		η = 1.0
		R₀ = mean(estimate_R₀(df, 0.1, γ, (1.0-i-r), i, r, (0, length(df[!, 1])))[1])

		return @dict(
			number_point_of_interest,
			migration_rate, T,
			R₀, γ, σ, ω, ξ, δ, η 
		)
	end

	function extract_params(df)
		e = 0.0/length(df[!, 1])
		i = df[1,:nuovi_positivi]/population
		r = df[1,:dimessi_guariti]/population
		d = df[1,:deceduti]/population
		s = (1.0-e-i-r-d)

		γ = 1.0/14 
		σ = 1.0/5.6 
		ω = 1.0/240 
		ξ = 0.0 
		δ = df[nrow(df), :deceduti] / sum(df[!, :nuovi_positivi]) 
		η = 1.0
		R₀ = mean(estimate_R₀(df, 0.1, γ, (1.0-i-r), i, r, (0, length(df[!, 1])))[1])

		u = [s, e, i, r, d] # scaled between [0-1]
		p = [R₀, γ, σ, ω, ξ, δ, η]
		return u, p, (0.0, length(df[!, 1]))
	end

	function save_parameters(params, path, title = title)
		isdir(path) == false && mkpath(path) 
		CSV.write(path*title*"_"*string(today())*".csv", DataFrame(params))
	end
end
