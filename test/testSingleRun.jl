using Test, Dates, DataFrames, Plots
include("../src/JEPMC.jl")

function save_results(path::String, p, d::DataFrame, plt::Plots.Plot)
    JEPMC.save_parameters(p, path, "SocialNetworkABM")
    JEPMC.save_dataframe(d, path, "SocialNetworkABM")
    JEPMC.save_plot(plt, path, "SocialNetworkABM")
end

function test_abm(path::String)
    model = JEPMC.init()
    data = JEPMC.collect!(model)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "no control")
    save_results(path * "singlerun/no_control/", model.properties, d, plt)
    return true
end

function test_abm_controller(path::String)
    model = JEPMC.init(; control = true)
    data = JEPMC.collect!(model)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "no pharmaceutical control")
    save_results(path * "singlerun/control/", model.properties, d, plt)
    return true
end

function test_abm_vaccine(path::String)
    model = JEPMC.init(; vaccine = true)
    data = JEPMC.collect!(model)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "pharmaceutical control")
    save_results(path * "singlerun/vaccine/", model.properties, d, plt)
    return true
end

function test_abm_all(path::String)
    model = JEPMC.init(; vaccine = true, control = true)
    data = JEPMC.collect!(model)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "all type of control")
    save_results(path * "singlerun/all/", model.properties, d, plt)
    return true
end
