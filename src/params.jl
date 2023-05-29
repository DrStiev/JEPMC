module model_params
using CSV, Random, Distributions, DataFrames
using DataFrames, DataDrivenDiffEq, DataDrivenSparse
using LinearAlgebra, OrdinaryDiffEq, ModelingToolkit
using Statistics, Downloads, DrWatson, Plots, Dates
using JLD2, FileIO, StableRNGs

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

function dataset_from_location(df::DataFrame, iso_code::String)
    df = filter(:iso_code => ==(iso_code), df)
    df[!, :total_susceptible] = df[!, :population] - df[!, :total_cases]
    return select(df, [:date]), select(
        df,
        [
            :new_cases_smoothed,
            :new_tests_smoothed,
            :new_vaccinations_smoothed,
            :new_deaths_smoothed,
        ],
    ), select(df, [:total_susceptible, :total_cases, :total_deaths, :total_tests]),
    select(df, [:reproduction_rate])
end

function read_local_dataset(path="data/OWID/owid-covid-data.csv")
    return DataFrame(CSV.File(path, delim=",", header=1))
end

# to be tested!
# https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/examples/example_02/
function system_identification(data::Array, ts, seed=1337)
    rng = StableRNG(seed)
    prob = ContinuousDataDrivenProblem(float.(data), float.(ts), GaussianKernel())

    len = size(data)[1]
    @variables u[1:len]
    h = polynomial_basis(u, 5)
    basis = Basis(h, u)
    println(basis)

    sampler = DataProcessing(split=0.8, shuffle=true, batchsize=30, rng=rng)
    # sparsity threshold
    λs = exp10.(-10:0.1:10)
    opt = STLSQ(λs) # iterate over different sparsity thresholds
    println("prob: $(size(prob)), basis: $(size(basis))")
    # DimensionMismatch: arrays could not be broadcast to a common size; got a dimension with lengths 5 and 0
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

# TODO: Uncertainti Quantified Deep Bayesian Model Discovery
# https://docs.sciml.ai/Overview/stable/showcase/bayesian_neural_ode/


function get_abm_parameters(C::Int, max_travel_rate::Float64, avg=1000; seed=1337)
    rng = Xoshiro(seed)
    pop = randexp(rng, C) * avg
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
    R₀ = 3.54
    # first(skipmissing(df[!, :reproduction_rate]))

    return @dict(number_point_of_interest, migration_rate, R₀, γ, σ, ω, ξ, δ, η, Rᵢ = 0.95,)
end

function get_ode_parameters(C::Int, avg=1000; seed=1337)
    rng = Xoshiro(seed)
    pop = randexp(rng, C) * avg
    number_point_of_interest = map((x) -> round(Int, x), pop)
    γ = 14 # infective period
    σ = 5 # exposed period
    ω = 280 # immunity period
    δ = 0.007
    R₀ = 3.54
    S = (sum(number_point_of_interest) - 1) / sum(number_point_of_interest)
    E = 0
    I = 1 / sum(number_point_of_interest)
    R = 0
    D = 0
    tspan = (1, 1200)
    return [S, E, I, R, D], [R₀, 1 / γ, 1 / σ, 1 / ω, δ], tspan
end

function save_parameters(params, path, title="parameters")
    isdir(path) == false && mkpath(path)
    save(path * title * ".jld2", params)
end

load_parameters(path) = load(path)
end
