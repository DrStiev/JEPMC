module CovidSim
using Pkg
Pkg.activate(".")
# Pkg.status()
# Pkg.update()
# Pkg.precompile()
# Pkg.instantiate()
# Pkg.resolve()
# Pkg.gc()

include("SocialNetworkABM.jl")
include("ABMUtils.jl")
include("Utils.jl")
include("Controller.jl")
# include("NeuralODE.jl")

export init, collect!, ensemble_collect!, collect_paramscan!, plot_system_graph
export plot_system_graph, plot_model
export save_dataframe, save_plot, save_parameters, read_dataset, load_parameters

# using JuliaFormatter
# format(".")
# using UpdateJulia
# update_julia() # to update julia version
end
