using BenchmarkTools
include("../src/JEPMC.jl")

# Benchmark tests
model = JEPMC.init()
@benchmark JEPMC.collect!(model)

model = JEPMC.init(; control = true)
@benchmark JEPMC.collect!(model)

model = JEPMC.init(; vaccine = true)
@benchmark JEPMC.collect!(model)

model = JEPMC.init(; control = true, vaccine = true)
@benchmark JEPMC.collect!(model)
