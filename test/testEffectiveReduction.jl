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
        models = [JEPMC.init(; control = c, vaccine = v, seed = abs(i))
                  for i in rand(Int64, numRuns)]
        data = JEPMC.ensemble_collect!(models)
        results[key] = data
    end
    plot_results(path, results)
    return true
end

function plot_results(path, results)
    plt = []
    for (i, (key, datas)) in enumerate(results)
        resm, resd, resh = 0.0, 0.0, 0.0
        for data in datas
            resm, resd, resh = [], [], []
            for d in data
                push!(resm, sum(d[!, :status]) ./ 50)
                push!(resd, d[!, :status][end][end])
                push!(resh, mean(d[!, :happiness]))
            end
            resm = mean(resm)
            resd = mean(resd)
            resh = mean(resh)
        end
        x = zeros(1, 4)
        rr = [resm[2], resm[3], resd, resh]
        x[1, :] = rr
        push!(plt,
            groupedbar(x,
                bar_position = :dodge,
                bar_width = 1.0,
                labels = ["E" "I" "D" "H"],
                title = key))
    end
    p = plot([plot(p, titlefontsize = 10) for (i, p) in enumerate(plt)]...,
        layout = (2, 2))
    JEPMC.save_plot(p, path, "reduction_comparison")
end
