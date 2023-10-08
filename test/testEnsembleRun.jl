using Test, Dates, DataFrames, Plots #, Distributed
# addprocs(Int(Sys.CPU_THREADS / 4))
#@everywhere include("../src/JEPMC.jl")
include("../src/JEPMC.jl")

function save_results(path::String, properties::Vector, d::DataFrame, plts::Vector)
    i = 1
    JEPMC.save_dataframe(d, path, "SocialNetworkABM")
    i = 1
    for plt in plts
        JEPMC.save_plot(plt, path, "SocialNetworkABM_$i")
        i += 1
    end
end

function test_ensemble_abm(path::String, control::Bool, vaccine::Bool)
    models = [JEPMC.init(; control = control, vaccine = vaccine, seed = abs(i))
              for i in rand(Int64, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d) for d in data]
    save_results(path, properties, d, plts)
    return true
end
