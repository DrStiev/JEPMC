module model_params
    using CSV, Random, Distributions, DataFrames
	using DataFrames, DataDrivenDiffEq, DataDrivenSparse
	using LinearAlgebra, OrdinaryDiffEq, ModelingToolkit
	using Statistics, Downloads, DrWatson, Plots

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

	function read_data(path="data/italy/dpc-covid19-ita-andamento-nazionale.csv")
		return DataFrame(CSV.File(path, delim=",", header=1))
	end

	# to be tested!
	function system_identification(data, seed=1234)
		prob = ContinuousDataDrivenProblem(Array(data)', 1:length(data[!,1]), GaussianKernel())
		# plot(prob) 

		# relevant states?
		@variables u[1:4]
		polys = Operation[]
		for i ∈ 0:3, j ∈ 0:3, k ∈ 0:3, l ∈ 0:3
			push!(polys, u[1]^i * u[2]^j * u[3]^k * u[4]^l)
		end
		h = [cos.(u)...; sin.(u)...; unique(polys)...]
		basis = Basis(h, u)

		sampler = DataProcessing(split=0.8, shuffle=true, batchsize=30, rng=Xoshiro(seed))
		# sparsity threshold
		λs = exp10.(-10:0.1:0)
		opt = STLSQ(λs) # iterate over different sparsity thresholds
		res = solve(prob, basis, opt, options=DataDrivenCommonOptions(data_processing=sampler, digits=1))

		#plot(plot(prob), plot(res), layout=(1,2))

		system = get_basis(res)
		params = get_parameter_map(system)

		return system, params
	end

	# find better estimator maybe from paper
	function estimate_R₀(data)
		return mean([data[i+1]/data[i] for i in 1:length(data)-1])
	end

	# find better estimation
	function estimate_control_growth(data)
		return (data[end,:tamponi]/data[1,:tamponi])^(1/length(data[!,1]))-1
	end

	function extract_params(df, C, max_travel_rate, avg=1000, population=58_850_717, seed=1234; outliers = [])
		rng = Xoshiro(seed)
		pop = randexp(rng, C) * avg # round(Int, (population / df[1,:nuovi_positivi]) / C) # alla fine non cambia il risultato
		pop = length(outliers) > 0 ? append!(pop, outliers) : pop
		C = length(outliers) > 0 ? C + length(outliers) : C
		number_point_of_interest = map((x) -> round(Int, x), pop)
		migration_rate = zeros(C, C)
		for c in 1:C
			for c2 in 1:C
				migration_rate[c,c2] = (number_point_of_interest[c] + number_point_of_interest[c2]) / number_point_of_interest[c]
			end
		end
		maxM = maximum(migration_rate)
		migration_rate = (migration_rate .* max_travel_rate) ./ maxM
		migration_rate[diagind(migration_rate)] .= 1.0

		γ = 14 # infective period
		σ = 5 # exposed period
		ω = 240 # immunity period
		ξ = 0.0 # vaccine ratio
		δ = df[nrow(df), :deceduti] / sum(df[!, :nuovi_positivi]) # mortality
		η = 1.0/20 # Countermeasures (social distancing, masks, etc...) (lower is better)
		ϵ = 1.0/10 # strong immune system
		θ = 0.0 # lockdown percentage
		θₜ = 90 # lockdown period
		q = 10 # quarantine period
		R₀ = estimate_R₀(df[!, :nuovi_positivi])
		ncontrols = df[1, :tamponi] / population
		control_growth = estimate_control_growth(df)
		# https://www.cochrane.org/CD013705/INFECTN_how-accurate-are-rapid-antigen-tests-diagnosing-covid-19#:~:text=In%20people%20with%20confirmed%20COVID,cases%20had%20positive%20antigen%20tests).
		# people with confirmed covid case (:I) -> (73 with symptoms + 55 no symptoms)/2 = 64% accuracy
		# people with confirmed covid case (:E) -> 82% accuracy
		# people with no covid (:S, :R) -> 99.7% accuracy 
		control_accuracy = [0.64, 0.82, 0.997]

		return @dict(
			number_point_of_interest, migration_rate, 
			ncontrols, control_growth, control_accuracy,
			R₀, γ, σ, ω, ξ, δ, η, ϵ, q, θ, θₜ,
		)
	end

	function extract_params(df, population = 58_850_717)
		e = 0.0/population
		i = df[1,:nuovi_positivi]/population
		r = df[1,:dimessi_guariti]/population
		d = df[1,:deceduti]/population
		s = (1.0-e-i-r-d)

		γ = 1/14 # infective period
		σ = 1/5 # exposed period
		ω = 1/240 # immunity period
		ξ = 0.0 # vaccine ratio
		δ = df[nrow(df), :deceduti] / sum(df[!, :nuovi_positivi]) # mortality
		η = 1.0/20 # Countermeasures (social distancing, masks, etc...) (lower is better)
		ϵ = 1.0/10 # strong immune system
		θ = 0.0 # lockdown percentage
		θₜ = 1/90 # lockdown period
		q = 1/10 # quarantine period
		R₀ = estimate_R₀(df[!, :nuovi_positivi])

		u = [s, e, i, r, d] # scaled between [0-1]
		p = [R₀, γ, σ, ω, ξ, δ, η, ϵ, q, θ, θₜ]
		return u, p, (0.0, length(df[!, 1]))
	end

	function save_parameters(params, path, title = title)
		isdir(path) == false && mkpath(path) 
		CSV.write(path*title*"_"*string(today())*".csv", DataFrame(params))
	end
end
