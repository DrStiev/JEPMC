using Test, Dates, BenchmarkTools

include("testSingleRun.jl")
@testset "singlerun" begin
    path = "results/" * string(today()) * "/"
    isdir(path) == false && mkpath(path)

    @test test_abm(path) == true
    @test test_abm_controller(path) == true
    @test test_abm_vaccine(path) == true
    @test test_abm_all(path) == true
end

include("testBenchmark.jl")
@testset "benchmark" begin
    path = "results/" * string(today()) * "/benchmark/"
    isdir(path) == false && mkpath(path)

    suite = BenchmarkGroup()
    suite["tag"] = BenchmarkGroup(["tag1", "tag2"])
    suite["no_control"] = @benchmarkable test_benchmark(false, false)
    suite["control"] = @benchmarkable test_benchmark(true, false)
    suite["vaccine"] = @benchmarkable test_benchmark(false, true)
    suite["all"] = @benchmarkable test_benchmark(true, true)

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
    path = "results/" * string(today()) * "/paramscanrun/"
    isdir(path) == false && mkpath(path)

    @test test_paramscan_abm(path * "maxTravelingRate/",
        Dict(:maxTravelingRate => Base.collect(0.001:0.003:0.01))) == true
    @test test_paramscan_abm(path * "edgesCoverage/",
        Dict(:edgesCoverage => [:high, :medium, :low])) == true
    @test test_paramscan_abm(path * "numNodes/", Dict(:numNodes => [8, 16, 32, 64])) == true
    @test test_paramscan_abm(path * "initialNodeInfected/",
        Dict(:initialNodeInfected => Base.collect(1:3:10))) == true
    @test test_paramscan_abm(path * "dt/",
        Dict(:dt => Base.collect(7:7:28), :control => [true])) == true
    @test test_paramscan_abm(path * "tolerance/",
        Dict(:tolerance => [1e-4, 1e-3, 1e-2, 1e-1], :control => [true])) == true
end

include("testEnsembleRun.jl")
@testset "ensemblerun" begin
    path = "results/" * string(today()) * "/"
    isdir(path) == false && mkpath(path)

    @test test_ensemble_abm(path) == true
    @test test_ensemble_abm_controller(path) == true
    @test test_ensemble_abm_vaccine(path) == true
    @test test_ensemble_abm_all(path) == true
end

include("testSensitivity.jl")
@testset "sensitivity" begin
    path = "results/" * string(today()) * "/sensitivity/"
    isdir(path) == false && mkpath(path)

    @test test_sensitivity(path) == true
end

include("testGif.jl")
@testset "gif_animation" begin
    path = "results/" * string(today()) * "/animation/"
    isdir(path) == false && mkpath(path)

    @test test_gif_animation_no_control(path) == true
    @test test_gif_animation_control(path) == true
    @test test_gif_animation_vaccine(path) == true
    @test test_gif_animation_all(path) == true
end

# include("testAqua.jl")
