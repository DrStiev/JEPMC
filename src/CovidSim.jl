module CovidSim
using Pkg #, JuliaFormatter, UpdateJulia
Pkg.activate(".")
# Pkg.status()
# Pkg.update()
# Pkg.gc()

include("SocialNetworkABM.jl")
include("ABMUtils.jl")
include("Utils.jl")

export init, collect, ensemble_collect
export plot_system_graph, plot_model
export save_dataframe, save_plot

# format(".")
# update_julia() # to update julia version
end
