using CSV, DataFrames, Downloads, Random, Plots
using JLD2, FileIO, Dates, Distributions
using LinearAlgebra: diagind
using DrWatson: @dict

function download_dataset(path::String, url::String)
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
    return select(df, [:date]),
    select(
        df,
        [
            :new_cases_smoothed,
            :new_tests_smoothed,
            :new_vaccinations_smoothed,
            :new_deaths_smoothed,
        ],
    ),
    select(df, [:total_susceptible, :total_cases, :total_deaths, :total_tests]),
    select(df, [:reproduction_rate])
end

function read_dataset(path::String)
    return DataFrame(CSV.File(path, delim=",", header=1))
end

function save_parameters(params, path::String, title::String="parameters")
    isdir(path) == false && mkpath(path)
    save(path * title * ".jld2", params)
end

load_parameters(path) = load(path)

function load_dataset(path::String)
    return DataFrame(CSV.File(path, delim=",", header=1))
end

function save_dataframe(data::DataFrame, path::String, title="StandardABM")
    isdir(path) == false && mkpath(path)
    CSV.write(path * title * "_" * string(today()) * ".csv", data)
end

function save_plot(plot, path="", title="title", format="png")
    isdir(path) == false && mkpath(path)
    savefig(plot, path * title * string(today()) * "." * format)
end

function split_dataset(data::DataFrame)
    res = [filter(:node => ==(i), data) for i in unique(data[!, :node])]
    return res
end

function split_and_plot(data::DataFrame)
    p1 = select(data, [:susceptible, :exposed, :infected, :recovered, :dead])
    p2 = select(data, [:active_countermeasures, :happiness])
    p3 = select(data, [:R0])
    l = @layout [
        RecipesBase.grid(1, 1)
        RecipesBase.grid(1, 2)
    ]
    p = plot(
        plot(
            Array(p1),
            labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
            title="ABM Dynamics",
        ),
        plot(Array(p2), labels=["η" "Happiness"], title="Agents response to η"),
        plot(Array(p3), labels="R₀", title="Reproduction number"),
        layout=l,
    )
    return p
end
