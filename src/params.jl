module model_params
    using CSV, Random, Distributions, DataFrames
	using DataFrames, DataDrivenDiffEq, DataDrivenSparse
	using LinearAlgebra, OrdinaryDiffEq, ModelingToolkit
	using Statistics, Downloads, DrWatson, Plots, Dates
	using JLD2, FileIO

	# https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
	# https://covid19.who.int/WHO-COVID-19-global-data.csv
	# https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
	# https://covid.ourworldindata.org/data/owid-covid-data.csv
	function download_dataset(path, url="https://covid.ourworldindata.org/data/owid-covid-data.csv")
		# https://github.com/owid/covid-19-data/tree/master/public/data/
		title = split(url,"/")
		isdir(path) == false && mkpath(path)
		return DataFrame(CSV.File(
			Downloads.download(url, path*title[length(title)]), 
			delim=",", header=1))
	end

	function dataset_from_location(df, iso_code)
		df = filter(:iso_code => ==(iso_code), df)
		return select!(df, [:date, :population, 
			:new_cases_smoothed, :new_deaths_smoothed, 
			:reproduction_rate, :new_tests_smoothed, 
			:new_vaccinations_smoothed])
	end

	function read_local_dataset(path="data/OWID/owid-covid-data.csv")
		return DataFrame(CSV.File(path, delim=",", header=1))
	end

	# to be tested!
	# https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
	function system_identification(data, seed=1337)
		# ERROR: InexactError: Int64(749.1302751376295)
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

	function get_abm_parameters(df, C, max_travel_rate, avg=1000; outliers=[], seed=1337)
		rng = Xoshiro(seed)
		pop = randexp(rng, C) * avg 
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
		ω = 280 # immunity period
		ξ = 0.0 # vaccine ratio
		δ = sum(skipmissing(df[!, :new_deaths_smoothed])) / 
			sum(skipmissing(df[!, :new_cases_smoothed])) # mortality
		η = 1.0/1 # Countermeasures (social distancing, masks, etc...) (lower is better)
		θ = 0.0 # lockdown percentage
		θₜ = 0 # lockdown period
		q = 10 # quarantine period
		R₀ = first(skipmissing(df[!, :reproduction_rate]))
		ncontrols = first(skipmissing(df[!, :new_tests_smoothed]))/df[1, :population]
		# https://www.cochrane.org/CD013705/INFECTN_how-accurate-are-rapid-antigen-tests-diagnosing-covid-19#:~:text=In%20people%20with%20confirmed%20COVID,cases%20had%20positive%20antigen%20tests).
		control_accuracy = [0.64, 0.82, 0.997]

		return @dict(
			number_point_of_interest, migration_rate, 
			ncontrols, control_accuracy,
			R₀, γ, σ, ω, ξ, δ, η, q, θ, θₜ,
		)
	end

	function save_parameters(params, path, title="parameters")
		isdir(path) == false && mkpath(path)
		save(path*title*".jld2", params) 
	end
end
