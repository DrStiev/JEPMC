using Test, Dates, Plots

include("../src/JEPMC.jl")

function test_gif_animation(path::String, control::Bool, vaccine::Bool)
    model = JEPMC.init(; control = control, vaccine = vaccine, numNodes = 20)
    data = JEPMC.collect!(model; n = 1, showprogress = false)
    anim = @animate for _ in 1:(1200 - 1)
        tmp = JEPMC.collect!(model; n = 1)
        for j in eachindex(tmp)
            push!(data[j], tmp[j][end, :])
        end
        plot(JEPMC.plot_system_graph(model), JEPMC.plot_model(data))
    end
    gif(anim, path, fps = 20)
    return true
end
