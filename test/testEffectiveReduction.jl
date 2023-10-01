using Test, Dates, DataFrames, Plots #, Distributed
using Statistics: mean
# addprocs(Int(Sys.CPU_THREADS / 4))
# @everywhere include("../src/JEPMC.jl")
include("../src/JEPMC.jl")

function test_run_and_plot(; numRuns = 10)
    control = [false, true]
    vaccine = [false, true]
    results = Dict()

    for c in control, v in vaccine
        key = "control: $(c), vaccine: $(v)"
        println("Running simulations for $key")
        data = []
        for _ in 1:numRuns
            model = JEPMC.init(; control = c, vaccine = v)
            push!(data, JEPMC.collect!(model))
        end
        results[key] = data
    end

    plot_results(results)
end

function plot_results(results)
    plt = plot(layout = (2, 2), legend = false)
    for (i, (key, data)) in enumerate(results)
        subplot = plt[i]
        for d in data
            m = mean([sum(d[!, :status]) for d in data]) ./ model.numNodes
            d = mean(([d[!, :status][end][end] for d in data]))
            h = mean([mean(d[!, :happiness]) for d in data])
            plot!(subplot, m[2], m[3], d, h, title = key)
        end
    end
    return plt
end
