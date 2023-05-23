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
function download_dataset(
    path,
    url="https://covid.ourworldindata.org/data/owid-covid-data.csv",
)
    # https://github.com/owid/covid-19-data/tree/master/public/data/
    title = split(url, "/")
    isdir(path) == false && mkpath(path)
    return DataFrame(
        CSV.File(
            Downloads.download(url, path * title[length(title)]),
            delim=",",
            header=1,
        ),
    )
end

function dataset_from_location(df, iso_code)
    df = filter(:iso_code => ==(iso_code), df)
    df[!, :total_susceptible] = df[!, :population] - df[!, :total_cases]
    return select(df, [:date]),
    mapcols(col -> replace(col, missing => 0), select(
        df,
        [
            :new_cases_smoothed,
            :new_tests_smoothed,
            :new_vaccinations_smoothed,
            :new_deaths_smoothed,
        ],
    )),
    mapcols(col -> replace(col, missing => 0), select(df, [:total_susceptible, :total_cases, :total_deaths, :total_tests])),
    mapcols(col -> replace(col, missing => 0), select(df, [:reproduction_rate]))
end

function read_local_dataset(path="data/OWID/owid-covid-data.csv")
    return DataFrame(CSV.File(path, delim=",", header=1))
end

# to be tested!
# https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
function system_identification(data, ts, seed=1337)
    prob = ContinuousDataDrivenProblem(
        float.(data), float.(ts), GaussianKernel(),
        U=(u, p, t) -> [exp(-((t - 5.0) / 5.0)^2)],
        p=ones(2))

    @variables u[1:size(data)[1]] c[1:1]
    @parameters w[1:size(data)[1]]
    u = collect(u)
    c = collect(c)
    w = collect(w)

    h = Num[cos.(u .* w); sin.(u .* w); polynomial_basis(u, 5); c]
    basis = Basis(h, u, parameters=w, controls=c)

    sampler = DataProcessing(split=0.8, shuffle=true, batchsize=30, rng=Xoshiro(seed))
    # sparsity threshold
    λs = exp10.(-10:0.1:0)
    opt = STLSQ(λs) # iterate over different sparsity thresholds
    res = solve(
        prob,
        basis,
        opt,
        options=DataDrivenCommonOptions(data_processing=sampler, digits=1),
    )
    system = get_basis(res)
    params = get_parameter_map(system)
    return system, params
end

function get_abm_parameters(C, max_travel_rate, avg=1000; outliers=[], seed=1337)
    rng = Xoshiro(seed)
    pop = randexp(rng, C) * avg
    pop = length(outliers) > 0 ? append!(pop, outliers) : pop
    C = length(outliers) > 0 ? C + length(outliers) : C
    number_point_of_interest = map((x) -> round(Int, x), pop)
    migration_rate = zeros(C, C)
    for c = 1:C
        for c2 = 1:C
            migration_rate[c, c2] =
                (number_point_of_interest[c] + number_point_of_interest[c2]) /
                number_point_of_interest[c]
        end
    end
    maxM = maximum(migration_rate)
    migration_rate = (migration_rate .* max_travel_rate) ./ maxM
    migration_rate[diagind(migration_rate)] .= 1.0

    γ = 14 # infective period
    σ = 5 # exposed period
    ω = 280 # immunity period
    ξ = 0.0 # vaccine ratio
    # https://www.nature.com/articles/s41467-021-22944-0
    δ = 0.007
    # sum(skipmissing(df[!, :new_deaths_smoothed])) / 
    # sum(skipmissing(df[!, :new_cases_smoothed])) # mortality
    η = 1.0 / 100 # Countermeasures speed
    θ = 0.0 # lockdown percentage
    θₜ = 0 # lockdown period
    q = 14 # quarantine period
    R₀ = 3.54
    # first(skipmissing(df[!, :reproduction_rate]))
    ncontrols = 0.00027#7.28E-5
    # first(skipmissing(df[!, :new_tests_smoothed]))/df[1, :population]
    # https://www.cochrane.org/CD013705/INFECTN_how-accurate-are-rapid-antigen-tests-diagnosing-covid-19#:~:text=In%20people%20with%20confirmed%20COVID,cases%20had%20positive%20antigen%20tests).
    control_accuracy = [0.775, 0.55, 0.997]

    return @dict(
        number_point_of_interest,
        migration_rate,
        ncontrols,
        control_accuracy,
        R₀,
        γ,
        σ,
        ω,
        ξ,
        δ,
        η,
        q,
        θ,
        θₜ,
        Rᵢ = 1.0,
    )
end

function get_ode_parameters(df)
    γ = 14 # infective period
    σ = 5 # exposed period
    ω = 280 # immunity period
    δ =
        sum(skipmissing(df[!, :new_deaths_smoothed])) /
        sum(skipmissing(df[!, :new_cases_smoothed])) # mortality
    R₀ = first(skipmissing(df[!, :reproduction_rate]))
    S = df[1, :population]
    E = 0
    I = 1
    R = 0
    D = 0
    tspan = (0, length(df[!, 1]))
    return [S, E, I, R, D], [R₀, γ, σ, ω, δ], tspan
end

function save_parameters(params, path, title="parameters")
    isdir(path) == false && mkpath(path)
    save(path * title * ".jld2", params)
end

load_parameters(path) = load(path)
end
