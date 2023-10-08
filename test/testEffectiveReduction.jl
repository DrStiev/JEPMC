using Test, Dates, DataFrames, Plots, StatsPlots #, Distributed
using Statistics: mean
# addprocs(Int(Sys.CPU_THREADS / 4))
# @everywhere include("../src/JEPMC.jl")
include("../src/JEPMC.jl")

function test_run_and_plot(path; numRuns = 10)
    control = [false, true]
    vaccine = [false, true]
    results = Dict()

    for c in control, v in vaccine
        key = "control: $(c), vaccine: $(v)"
        println("Running simulations for $key")
        models = [JEPMC.init(; control = c, vaccine = v, seed = abs(i))
                  for i in rand(Int64, numRuns)]
        data = JEPMC.ensemble_collect!(models)
        results[key] = data
    end
    results
    plot_results(path, results)
end

function plot_results(path, results)
    plt = []
    for (i, (key, datas)) in enumerate(results)
        println("index: $i, key: $key, data: $(size(data[1]))")
        resm = mean([sum(d[!, :status]) for d in data] for data in datas)
        resd = mean([d[!, :status][end][end] for d in data] for data in datas)
        resh = mean(mean([d[!, :happiness] for d in data]) for data in datas)
        r = resm ./ 50
        rr = [r[2], r[3], resh, resd]
        push!(plt,
            groupedbar(rr,
                bar_position = :dodge,
                bar_width = 1.0,
                labels = ["AVG CUM E" "AVG CUM I" "AVG H" "AVG TOT D"],
                title = key))
    end
    p = plot([plot(p, titlefontsize = 10) for (i, p) in enumerate(plt)]...,
        layout = (2, 2))
    JEPMC.save_plot(p, path, "reduction_comparison")
    return true
end
