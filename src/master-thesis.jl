using Pkg
using JuliaFormatter
Pkg.activate(".")
format(".")
Pkg.update()
Pkg.resolve()
Pkg.precompile()
Pkg.instantiate()
