using Pkg, JuliaFormatter
Pkg.activate(".")
format(".")

Pkg.status()
Pkg.update()
Pkg.resolve()
Pkg.precompile()
Pkg.instantiate()
Pkg.gc()
