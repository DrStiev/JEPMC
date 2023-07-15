# using CovidSim
using Test

include("../src/CovidSim.jl")

@testset "CovidSim.jl" begin
    @test test_abm() == true
    @test test_abm_controller() == true
    @test test_ensemble_abm() == true
    @test test_ensemble_abm_controller() == true
    @test test_paramscan_abm() == true
end

function test_abm()
    model = CovidSim.init()
    CovidSim.save_parameters(
        model.properties,
        "data/abm/no_control/",
        "SocialNetworkABM_NO_CONTROL"
    )
    data = CovidSim.collect!(model)
    d = reduce(vcat, data)
    CovidSim.save_dataframe(
        d,
        "data/abm/no_control/",
        "SocialNetworkABM_NO_CONTROL"
    )
    plt = CovidSim.plot_model(data)
    CovidSim.save_plot(
        plt,
        "img/abm/no_control/",
        "SocialNetworkABM_NO_CONTROL",
        "pdf"
    )
    return true
end

function test_abm_controller()
    model = CovidSim.init(; control=true)
    CovidSim.save_parameters(
        model.properties,
        "data/abm/control/",
        "SocialNetworkABM_CONTROL"
    )
    data = CovidSim.collect!(model)
    d = reduce(vcat, data)
    CovidSim.save_dataframe(
        d,
        "data/abm/control/",
        "SocialNetworkABM_CONTROL"
    )
    plt = CovidSim.plot_model(data)
    CovidSim.save_plot(
        plt,
        "img/abm/control/",
        "SocialNetworkABM_CONTROL",
        "pdf"
    )
    return true
end

function test_ensemble_abm()
    models = [CovidSim.init(; seed=abs(i)) for i in rand(Int64, 10)]
    i = 1
    for model in models
        CovidSim.save_parameters(
            model.properties,
            "data/abm/ensemble/no_control/",
            "SocialNetworkABM_ENSEMBLE_$i"
        )
    end
    data = CovidSim.collect!(model)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    CovidSim.save_dataframe(
        d,
        "data/abm/ensemble/no_control/",
        "SocialNetworkABM_ENSEMBLE"
    )
    i = 1
    for d in data
        plt = CovidSim.plot_model(data)
        CovidSim.save_plot(
            plt,
            "img/abm/ensemble/no_control/",
            "SocialNetworkABM_ENSEMBLE_$i",
            "pdf"
        )
        i += 1
    end
    return true
end

function test_ensemble_abm_controller()
    models = [CovidSim.init(; control=true, seed=abs(i)) for i in rand(Int64, 10)]
    i = 1
    for model in models
        CovidSim.save_parameters(
            model.properties,
            "data/abm/ensemble/control/",
            "SocialNetworkABM_ENSEMBLE_CONTROL_$i"
        )
    end
    data = CovidSim.collect!(model)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    CovidSim.save_dataframe(
        d,
        "data/abm/ensemble/control/",
        "SocialNetworkABM_ENSEMBLE_CONTROL"
    )
    i = 1
    for d in data
        plt = CovidSim.plot_model(data)
        CovidSim.save_plot(
            plt,
            "img/abm/ensemble/control/",
            "SocialNetworkABM_ENSEMBLE_CONTROL_$i",
            "pdf"
        )
        i += 1
    end
    return true
end

function test_paramscan_abm()
    parameters = Dict(
        :maxTravelingRate => Base.collect(0.01:0.01:0.1),
        :edgesCoverage => [:high, :medium, :low],
        :numNodes => Base.collect(1:10:101),
        :avgPopulation => Base.collect(1000:1000:100000),
        :control => [true, false],
    )

    data = CovidSim.collect_paramscan!(parameters, CovidSim.init())
    d = reduce(vcat, data)
    CovidSim.save_dataframe(
        d,
        "data/abm/paramscan/",
        "SocialNetworkABM_PARAMSCAN"
    )
    return true
end
