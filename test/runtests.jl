using Test

include("testBenchmark.jl")
@testset "benchmark" begin
    path = "results/" * string(today()) * "/benchmark/"
    isdir(path) == false && mkpath(path)

    @benchmark test_benchmark(false, false)
    @benchmark test_benchmark(false, true)
    @benchmark test_benchmark(true, false)
    @benchmark test_benchmark(true, true)
end

include("testSingleRun.jl")
@testset "singlerun" begin
    path = "results/" * string(today()) * "/"
    isdir(path) == false && mkpath(path)

    @test test_abm(path) == true
    @test test_abm_controller(path) == true
    @test test_abm_vaccine(path) == true
    @test test_abm_all(path) == true
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
    @test test_paramscan_abm(path * "dt/", Dict(:dt => Base.collect(7:7:28))) == true
    @test test_paramscan_abm(path * "tolerance/",
        Dict(:tolerance => [1e-4, 1e-3, 1e-2, 1e-1])) == true
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
