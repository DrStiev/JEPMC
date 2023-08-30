# using JEPMC
using Distributed
addprocs(Int(Sys.CPU_THREADS / 4))
@everywhere include("../src/JEPMC.jl")

using Test, Dates, DataFrames, Plots
include("../src/JEPMC.jl")

function save_results(path::String, p, d::DataFrame, plt::Plots.Plot)
    JEPMC.save_parameters(p, path * "parameters/", "SocialNetworkABM")
    JEPMC.save_dataframe(d, path * "dataframe/", "SocialNetworkABM")
    JEPMC.save_plot(plt, path * "plot/", "SocialNetworkABM", "pdf")
end

function save_results(path::String, properties::Vector, d::DataFrame, plts::Vector)
    i = 1
    for p in properties
        JEPMC.save_parameters(p, path * "parameters/", "SocialNetworkABM_$i")
        i += 1
    end
    JEPMC.save_dataframe(d, path * "dataframe/", "SocialNetworkABM")
    i = 1
    for plt in plts
        JEPMC.save_plot(plt, path * "plot/", "SocialNetworkABM_$i", "pdf")
        i += 1
    end
end

function test_abm(path::String)
    model = JEPMC.init()
    data = JEPMC.collect!(model; showprogress = true)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "no control")
    save_results(path * "singlerun/no_control/", model.properties, d, plt)
    return true
end

function test_abm_controller(path::String)
    model = JEPMC.init(; control = true)
    data = JEPMC.collect!(model; showprogress = true)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "no pharmaceutical control")
    save_results(path * "singlerun/control/", model.properties, d, plt)
    return true
end

function test_abm_vaccine(path::String)
    model = JEPMC.init(; vaccine = true)
    data = JEPMC.collect!(model; showprogress = true)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "pharmaceutical control")
    save_results(path * "singlerun/vaccine/", model.properties, d, plt)
    return true
end

function test_abm_all(path::String)
    model = JEPMC.init(; vaccine = true, control = true)
    data = JEPMC.collect!(model; showprogress = true)
    d = reduce(vcat, data)
    plt = JEPMC.plot_model(data; errorstyle = :ribbon, title = "all type of control")
    save_results(path * "singlerun/all/", model.properties, d, plt)
    return true
end

@testset "singlerun" begin
    path = "results/" * string(today()) * "/"
    isdir(path) == false && mkpath(path)

    @test test_abm(path) == true
    @test test_abm_controller(path) == true
    @test test_abm_vaccine(path) == true
    @test test_abm_all(path) == true
end

function test_gif_animation_no_control(path::String)
    model = JEPMC.init(; numNodes = 20)
    data = JEPMC.collect!(model; n = 1)
    anim = @animate for _ in 1:(1200 - 1)
        tmp = JEPMC.collect!(model; n = 1)
        for j in 1:size(tmp, 1)
            push!(data[j], tmp[j][end, :])
        end
        plot(JEPMC.plot_system_graph(model), JEPMC.plot_model(data))
    end
    gif(anim, path * "demo_no_control.gif", fps = 20)
    return true
end

function test_gif_animation_control(path::String)
    model = JEPMC.init(; numNodes = 20, control = true)
    data = JEPMC.collect!(model; n = 1)
    anim = @animate for _ in 1:(1200 - 1)
        tmp = JEPMC.collect!(model; n = 1)
        for j in 1:size(tmp, 1)
            push!(data[j], tmp[j][end, :])
        end
        plot(JEPMC.plot_system_graph(model), JEPMC.plot_model(data))
    end
    gif(anim, path * "demo_control.gif", fps = 20)
    return true
end

function test_gif_animation_vaccine(path::String)
    model = JEPMC.init(; numNodes = 20, vaccine = true)
    data = JEPMC.collect!(model; n = 1)
    anim = @animate for _ in 1:(1200 - 1)
        tmp = JEPMC.collect!(model; n = 1)
        for j in 1:size(tmp, 1)
            push!(data[j], tmp[j][end, :])
        end
        plot(JEPMC.plot_system_graph(model), JEPMC.plot_model(data))
    end
    gif(anim, path * "demo_vaccine.gif", fps = 20)
    return true
end

function test_gif_animation_all(path::String)
    model = JEPMC.init(; numNodes = 20, control = true, vaccine = true)
    data = JEPMC.collect!(model; n = 1)
    anim = @animate for _ in 1:(1200 - 1)
        tmp = JEPMC.collect!(model; n = 1)
        for j in 1:size(tmp, 1)
            push!(data[j], tmp[j][end, :])
        end
        plot(JEPMC.plot_system_graph(model), JEPMC.plot_model(data))
    end
    gif(anim, path * "demo_all_control.gif", fps = 20)
    return true
end

@testset "gif_animation" begin
    path = "results/" * string(today()) * "/animation/"
    isdir(path) == false && mkpath(path)

    @test test_gif_animation_no_control(path) == true
    @test test_gif_animation_control(path) == true
    @test test_gif_animation_vaccine(path) == true
    @test test_gif_animation_all(path) == true
end

function test_ensemble_abm(path::String)
    models = [JEPMC.init(; seed = Int64(abs(i))) for i in rand(Int8, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models; showprogress = true)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/no_control/", properties, d, plts)
    return true
end

function test_ensemble_abm_controller(path::String)
    models = [JEPMC.init(; control = true, seed = Int64(abs(i))) for i in rand(Int8, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models; showprogress = true)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/control/", properties, d, plts)
    return true
end

function test_ensemble_abm_vaccine(path::String)
    models = [JEPMC.init(; vaccine = true, seed = Int64(abs(i))) for i in rand(Int8, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models; showprogress = true)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/vaccine/", properties, d, plts)
    return true
end

function test_ensemble_abm_all(path::String)
    models = [JEPMC.init(; control = true, vaccine = true, seed = Int64(abs(i))) for
              i in rand(Int8, 5)]
    properties = [model.properties for model in models]
    data = JEPMC.ensemble_collect!(models; showprogress = true)
    d = reduce(vcat, data)
    d = reduce(vcat, d)
    plts = [JEPMC.plot_model(d; errorstyle = :ribbon) for d in data]
    save_results(path * "ensemblerun/all/", properties, d, plts)
    return true
end

@testset "ensemblerun" begin
    path = "results/" * string(today()) * "/"
    isdir(path) == false && mkpath(path)

    @test test_ensemble_abm(path) == true
    @test test_ensemble_abm_controller(path) == true
    @test test_ensemble_abm_vaccine(path) == true
    @test test_ensemble_abm_all(path) == true
end

function complex_filter(x, y, z, w, k, j, h)
    x == val[1] &&
        y == val[2] &&
        z == val[3] &&
        w == val[4] &&
        k == val[5] &&
        j == val[6] &&
        h == val[7]
end

function complex_filter(x, y, z, w, k, j)
    x == val[1] && y == val[2] && z == val[3] && w == val[4] && k == val[5] && j == val[6]
end

function complex_filter(x, y, z, w, k)
    x == val[1] && y == val[2] && z == val[3] && w == val[4] && k == val[5]
end

function complex_filter(x, y, z, w)
    x == val[1] && y == val[2] && z == val[3] && w == val[4]
end

function complex_filter(x, y, z)
    x == val[1] && y == val[2] && z == val[3]
end

function complex_filter(x, y)
    x == val[1] && y == val[2]
end

function complex_filter(x)
    x == val[1]
end

val = nothing

function test_paramscan_abm(path::String, properties)
    data = JEPMC.collect_paramscan!(properties; showprogress = true)
    plts = []
    global val = nothing
    for i in 1:size(data[2], 1)
        namesz = names(data[2][i, :])
        global val = data[2][i, :]
        # hardcoded but functional
        df = filter(namesz => complex_filter, data[1])
        select!(df, Not(namesz))
        dd = [filter(:id => ==(i), df) for i in unique(df[!, :id])]
        r = []
        for i in 1:size(names(val), 1)
            push!(r, names(val)[i] * ": " * string(val[i]))
        end
        j = join(r, ", ")
        push!(plts, JEPMC.plot_model(dd; title = j))
    end
    JEPMC.save_dataframe(data[1], path * "dataframe/", "SocialNetworkABM")
    i = 1
    for plt in plts
        JEPMC.save_plot(plt, path * "plot/", "SocialNetworkABM_$i", "pdf")
        i += 1
    end
    return true
end

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

function test_sensitivity(path::String)
    x, dp = JEPMC.sensitivity_analisys(seir!,
        [0.999, 0.0, 0.001, 0.0, 0.0],
        (0.0, 1200.0),
        [3.54, 1 / 14, 1 / 5, 1 / 280, 0.001, 0.0, 0.0])
    JEPMC.save_plot(pltout, path, "sensitivity_analysis", "pdf")
    return true
end

@testset "sensitivity" begin
    path = "results/" * string(today()) * "/sensitivity/"
    isdir(path) == false && mkpath(path)

    @test test_sensitivity(path) == true
end
