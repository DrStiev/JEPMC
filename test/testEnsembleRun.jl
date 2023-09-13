using Test, Dates, DataFrames, Plots #, Distributed
# addprocs(Int(Sys.CPU_THREADS / 4))
#@everywhere include("../src/JEPMC.jl")
include("../src/JEPMC.jl")

function save_results(path::String, properties::Vector, d::DataFrame, plts::Vector)
    i = 1
    for p in properties
        JEPMC.save_parameters(p, path, "SocialNetworkABM_$i")
        i += 1
    end
    JEPMC.save_dataframe(d, path, "SocialNetworkABM")
    i = 1
    for plt in plts
        JEPMC.save_plot(plt, path, "SocialNetworkABM_$i")
        i += 1
    end
end

function test_ensemble_abm(path::String)
    models = [JEPMC.init(; seed = abs(i)) for i in rand(Int64, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/no_control/", properties, d, plts)
    return true
end

function test_ensemble_abm_controller(path::String)
    models = [JEPMC.init(; control = true, seed = abs(i)) for i in rand(Int64, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/control/", properties, d, plts)
    return true
end

function test_ensemble_abm_vaccine(path::String)
    models = [JEPMC.init(; vaccine = true, seed = abs(i)) for i in rand(Int64, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/vaccine/", properties, d, plts)
    return true
end

function test_ensemble_abm_all(path::String)
    models = [JEPMC.init(; control = true, vaccine = true, seed = abs(i)) for
              i in rand(Int64, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/all/", properties, d, plts)
    return true
end
