# https://julialang.org/contribute/developing_package/
module CovidSim
using Pkg #, JuliaFormatter, UpdateJulia
Pkg.activate(".")
# Pkg.status()
# Pkg.update()
# Pkg.gc()

include("ABM.jl")
include("ABMUtilis.jl")
include("Controller.jl")
include("ControllerUtils.jl")
include("Utils.jl")

export init, collect, ensemble_collect, save_dataframe, save_plot

# format(".")
# update_julia() # to update julia version
end
