### -*- Mode: Julia -*-

### Utils.jl
###
### See file LICENSE in top folder for copyright and licensing
### information.

using CSV, DataFrames, Downloads, Random, Plots
using JLD2, FileIO, Dates, Distributions
using LinearAlgebra: diagind
using DrWatson: @dict

function ensure_directory_exists(path::String)
    isdir(path) == false && mkpath(path)
end

"""
    download_dataset(path::String, url::String)
    Download dataset from an URL string into a specific path
"""
function download_dataset(path::String, url::String)
    title = split(url, "/")
    ensure_directory_exists(path)
    return DataFrame(CSV.File(Downloads.download(url, joinpath(path, title[end])),
        delim = ",",
        header = 1))
end

function dataset_from_location(df::DataFrame, iso_code::String)
    df = filter(:iso_code => ==(iso_code), df)
    df[!, :total_susceptible] = df[!, :population] - df[!, :total_cases]
    return select(df, [:date]),
    select(df,
        [
            :new_cases_smoothed,
            :new_tests_smoothed,
            :new_vaccinations_smoothed,
            :new_deaths_smoothed,
        ]),
    select(df, [:total_susceptible, :total_cases, :total_deaths, :total_tests]),
    select(df, [:reproduction_rate])
end

"""
    read_dataset(path::String)
    Read a dataset from a specific path and return a DataFrame Object
"""
function read_dataset(path::String)
    return DataFrame(CSV.File(path, delim = ",", header = 1))
end

"""
    save_parameters(params::Dict, path::String, [title::String])
    Save a dictionary of parameters into a specific path. If no title is provided then a default one is used.
"""
function save_parameters(params, path::String, title::String = "parameters")
    ensure_directory_exists(path)
    save(joinpath(path, "$title.jld2"), params)
end

"""
    load_parameters(path::String)
    Load a dictionary of parameters from a specific path
"""
load_parameters(path) = load(path)

"""
    save_dataframe(data::DataFrame, path::String, [title::String])
    Save a DataFrame into a specific path. If no title is provided then a default one is used.
"""
function save_dataframe(data::DataFrame, path::String, title = "StandardABM")
    ensure_directory_exists(path)
    CSV.write(joinpath(path, "$title.csv"), data)
end

"""
    save_plot(plot, path::String, [title::String], [format::String])
    Save a plot into a specific path. If no title is provided then a default one is used. If no format is specified then .png is used
"""
function save_plot(plot, path = "", title = "title", format = "png")
    ensure_directory_exists(path)
    savefig(plot, joinpath(path, "$title.$format"))
end

### end of file -- Utils.jl
