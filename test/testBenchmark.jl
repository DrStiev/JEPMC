using BenchmarkTools
include("../src/JEPMC.jl")

function test_benchmark(control, vaccine)
    model = JEPMC.init(; control = control, vaccine = vaccine)
    JEPMC.collect!(model; showprogress=false)
end
