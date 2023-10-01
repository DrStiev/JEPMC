using Test, Dates, DataFrames, Plots
include("../src/JEPMC.jl")

function save_results(path::String, p, d::DataFrame, plt::Plots.Plot)
    JEPMC.save_parameters(p, path, "SocialNetworkABM")
    JEPMC.save_dataframe(d, path, "SocialNetworkABM")
    JEPMC.save_plot(plt, path, "SocialNetworkABM")
end

function test_abm(path::String, control::Bool, vaccine::Bool)
    model = JEPMC.init(; control = control, vaccine = vaccine)
    data = JEPMC.collect!(model)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "no control")
    save_results(path, model.properties, d, plt)
    return true
end
