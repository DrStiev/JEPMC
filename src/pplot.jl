module pplot
    using Plots, LaTeXStrings, StatsPlots
    using InteractiveDynamics, CairoMakie
    using DataFrames, SciMLBase, Dates, CSV

    function static_preplot!(ax, model)
        CairoMakie.hidedecorations!(ax)
        for l in 1:length(model.attr_pos)
            # mostro posizione attrattore
            obj = CairoMakie.scatter!([model.attr_pos[l][1] model.attr_pos[l][2]]; color = :blue)
            CairoMakie.translate!(obj, 0, 0, 5)
        end
    end

    # different epidemic states: S, E, I, R
    colors(a) = a.status == :S ? "grey80" : a.status == :E ? "yellow" : a.status == :I ? "red" : "green"

    function record_video(model, astep, mstep;
        name = "img/sngraph_", 
        framerate = 15, frames = 100, title = "title")
        name = name*string(today())*".mp4"
        abmvideo(
            name, model, astep, mstep;
            framerate=framerate, frames=frames, 
            title=title, static_preplot!, ac = colors,
        )
    end

    function line_plot(data::DataFrame, title = "title")
        p = @df data Plots.plot(cols(), title = title, lw = 2, xlabel = L"Days")
        savefig(p, "img/"*title*"_"*string(today())*".png")
    end

    function save_parameters(model, title = title)
        df = DataFrame(model.properties)
        CSV.write("data/"*title*"_"*string(today()), df)
    end
end