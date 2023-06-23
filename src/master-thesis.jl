# https://julialang.org/contribute/developing_package/

using Pkg #, JuliaFormatter, UpdateJulia
Pkg.activate(".")
# format(".")
# update_julia() # to update julia version

Pkg.status()
Pkg.update()
Pkg.resolve()
Pkg.precompile()
Pkg.instantiate()
Pkg.gc()
