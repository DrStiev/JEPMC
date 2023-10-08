using Test, Dates, BenchmarkTools

function create_path(subfolder = "")
    path = "results/" * string(today()) * "/" * subfolder
    isdir(path) == false && mkpath(path)
    return path
end

include("testSingleRun.jl")
@testset "singlerun" begin
    path = create_path("singlerun/")

    control = [false, true]
    vaccine = [false, true]

    for c in control, v in vaccine
        spath = path * (c ? "control_" : "") * (v ? "vaccine_" : "") * "/"
        @test test_abm(spath, c, v)
    end
end

include("testBenchmark.jl")
@testset "benchmark" begin
    path = create_path("benchmark/")
    suite = BenchmarkGroup()
    benchmarks = Dict("no_control" => (false, false),
        "control" => (true, false),
        "vaccine" => (false, true),
        "all" => (true, true))

    for (key, value) in benchmarks
        suite[key] = @benchmarkable test_benchmark(value...)
    end

    tune!(suite)
    res = BenchmarkTools.run(suite, verbose = true)
    BenchmarkTools.save(path * "benchmark.json", res)
end

include("testParamScan.jl")
@testset "paramscanrun" begin
    properties = Dict(:maxTravelingRate => Base.collect(0.001:0.003:0.01),
        :edgesCoverage => [:high, :medium, :low],
        :numNodes => [8, 16, 32, 64],
        :initialNodeInfected => Base.collect(1:3:10),
        :dt => Base.collect(7:7:28),
        :tolerance => [1e-4, 1e-3, 1e-2, 1e-1])
    path = create_path("paramscan/")

    for (key, value) in properties
        @test test_paramscan_abm(path * string(key) * "/", Dict(key => value))
    end
end

include("testEnsembleRun.jl")
@testset "ensemblerun" begin
    path = create_path("ensemblerun/")

    control = [false, true]
    vaccine = [false, true]

    for c in control, v in vaccine
        spath = path * (c ? "control_" : "") * (v ? "vaccine_" : "") * "/"
        @test test_ensemble_abm(spath, c, v)
    end
end

include("testSensitivity.jl")
@testset "sensitivity" begin
    path = create_path("sensitivity/")

    @test test_sensitivity(path)
end

include("testEffectiveReduction.jl")
@testset "reduction_effectiveness" begin
    path = create_path("intervention_effectiveness")

    @test test_run_and_plot(path; numRuns = 5)
end

include("testGif.jl")
@testset "gif_animation" begin
    path = create_path("animation/")

    control = [false, true]
    vaccine = [false, true]

    for c in control, v in vaccine
        spath = path * (c ? "control_" : "") * (v ? "vaccine_" : "") * ".gif"
        @test test_gif_animation(spath, c, v)
    end
end

# include("testAqua.jl")

### end of file -- runtests.jl
