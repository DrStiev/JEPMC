# using CovidSim
using Test, Dates

include("../src/CovidSim.jl")

# add test with 8000 nodes and avgPopulation=7000 (w/ and wout/ control)
@testset "CovidSim.jl" begin
    @test test_abm() == true
    @test test_abm_controller() == true
    @test test_ensemble_abm() == true
    @test test_ensemble_abm_controller() == true # requires lot of time (4+ hours)
    @test test_paramscan_abm() == true # requires lot of time (4+ hours)
end

function test_abm()
    model = CovidSim.init()
    CovidSim.save_parameters(
        model.properties,
        "data/abm/no_control/parameters/" * string(today()) * "/",
        "SocialNetworkABM_NO_CONTROL",
    )
    data = CovidSim.collect!(model)
    d = reduce(vcat, data)
    CovidSim.save_dataframe(
        d,
        "data/abm/no_control/dataframe/" * string(today()) * "/",
        "SocialNetworkABM_NO_CONTROL",
    )
    plt = CovidSim.plot_model(data)
    CovidSim.save_plot(
        plt,
        "img/abm/no_control/" * string(today()) * "/",
        "SocialNetworkABM_NO_CONTROL",
        "pdf",
    )
    return true
end

function test_abm_controller()
    model = CovidSim.init(; control = true)
    CovidSim.save_parameters(
        model.properties,
        "data/abm/control/parameters/" * string(today()) * "/",
        "SocialNetworkABM_CONTROL",
    )
    data = CovidSim.collect!(model)
    d = reduce(vcat, data)
    CovidSim.save_dataframe(
        d,
        "data/abm/control/dataframe/" * string(today()) * "/",
        "SocialNetworkABM_CONTROL",
    )
    plt = CovidSim.plot_model(data)
    CovidSim.save_plot(
        plt,
        "img/abm/control/" * string(today()) * "/",
        "SocialNetworkABM_CONTROL",
        "pdf",
    )
    return true
end

function test_ensemble_abm()
    models = [CovidSim.init(; seed = abs(i)) for i in rand(Int8, 10)]
    i = 1
    for model in models
        CovidSim.save_parameters(
            model.properties,
            "data/abm/ensemble/no_control/parameters/" * string(today()) * "/",
            "SocialNetworkABM_ENSEMBLE_$i",
        )
        i += 1
    end
    data = CovidSim.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    CovidSim.save_dataframe(
        d,
        "data/abm/ensemble/no_control/dataframe/" * string(today()) * "/",
        "SocialNetworkABM_ENSEMBLE",
    )
    i = 1
    for d in data
        plt = CovidSim.plot_model(d)
        CovidSim.save_plot(
            plt,
            "img/abm/ensemble/no_control/" * string(today()) * "/",
            "SocialNetworkABM_ENSEMBLE_$i",
            "pdf",
        )
        i += 1
    end
    return true
end

function test_ensemble_abm_controller()
    models = [CovidSim.init(; control = true, seed = abs(i)) for i in rand(Int8, 10)]
    i = 1
    for model in models
        CovidSim.save_parameters(
            model.properties,
            "data/abm/ensemble/control/parameters/" * string(today()) * "/",
            "SocialNetworkABM_ENSEMBLE_CONTROL_$i",
        )
        i += 1
    end
    data = CovidSim.ensemble_collect!(models)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    CovidSim.save_dataframe(
        d,
        "data/abm/ensemble/control/dataframe/" * string(today()) * "/",
        "SocialNetworkABM_ENSEMBLE_CONTROL",
    )
    i = 1
    for d in data
        plt = CovidSim.plot_model(d)
        CovidSim.save_plot(
            plt,
            "img/abm/ensemble/control/" * string(today()) * "/",
            "SocialNetworkABM_ENSEMBLE_CONTROL_$i",
            "pdf",
        )
        i += 1
    end
    return true
end

function test_paramscan_abm()
    data = CovidSim.collect_paramscan!()
    d = reduce(vcat, data)
    CovidSim.save_dataframe(
        d,
        "data/abm/paramscan/" * string(today()) * "/",
        "SocialNetworkABM_PARAMSCAN",
    )
    return true
end
