using Test, Dates, Plots
include("../src/SocialNetworkABM.jl")

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
