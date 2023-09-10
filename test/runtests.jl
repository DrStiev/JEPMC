using Test
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
        :numNodes => Base.collect(5:25:80),
        :initialNodeInfected => Base.collect(1:3:10))
    path = "results/" * string(today()) * "/paramscanrun/"
    isdir(path) == false && mkpath(path)

    @test test_paramscan_abm(path * "maxTravelingRate/",
        Dict(:maxTravelingRate => Base.collect(0.001:0.003:0.01))) == true
    @test test_paramscan_abm(path * "edgesCoverage/",
        Dict(:edgesCoverage => [:high, :medium, :low])) == true
    @test test_paramscan_abm(path * "numNodes/",
        Dict(:numNodes => Base.collect(5:25:80))) == true
    @test test_paramscan_abm(path * "initialNodeInfected/",
        Dict(:initialNodeInfected => Base.collect(1:3:10))) == true
end

@testset "controller_paramscan" begin
    path = "results/" * string(today()) * "/controller_paramscan/"
    isdir(path) == false && mkpath(path)

    for i in [1e-4, 1e-3, 1e-2, 1e-1]
        for j in 7:7:28
            control_options = Dict(:tolerance => i,
                :dt => j,
                :step => 3.0,
                :maxiters => 10,
                :patience => 3,
                :doplot => false,
                :loss => missing,
                :υ_max => missing)
            @test test_param_controller(control_options, "tolerance $i, dt $j") == true
        end
    end
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

include("testBenchmark.jl")
include("testAqua.jl")
