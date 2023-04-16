module pplot
    using Plots, LaTeXStrings, StatsPlots
    using InteractiveDynamics, CairoMakie
    using DataFrames, SciMLBase, Dates, CSV
    using Agents

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

    function record_video(abmobs, model, name="title", n=100)
        infected_fraction(m, x) = count(m[id].status == :I for id in x) / length(x)
        infected_fractions(m) = [infected_fraction(m, ids_in_position(p, m)) for p in positions(m)]
        fracs = lift(infected_fractions, abmobs.model)
        color = lift(fs -> [cgrad(:inferno)[f] for f in fs], fracs)
        title = lift(
            (s, m) -> "step = $(s), infected = $(round(Int, 100infected_fraction(m, allids(m))))%",
            abmobs.s, abmobs.model
        ) 

        fig = Figure(resolution = (600, 400))
        ax = Axis(fig[1, 1]; title, xlabel = "City", ylabel = "Population")
        barplot!(ax, model.number_point_of_interest; strokecolor = :black, strokewidth = 1, color)

        record(fig, "img/"*name*".mp4"; framerate = 5) do io
            for j in 1:n
                recordframe!(io)
                Agents.step!(abmobs, 1)
            end
            recordframe!(io)
        end
    end

    function line_plot(data, timeperiod, path="", title = "title", format="png")
        dates = range(timeperiod[1], timeperiod[end], step=Day(1))
	    tm_ticks = round.(dates, Month(1)) |> unique;
        p = Plots.plot(timeperiod, Matrix(data), labels=permutedims(names(data)), 
            title=title, xticks=(tm_ticks, Dates.format.(tm_ticks, "uu/yyyy")), 
            xrot=45, xminorticks=true, xlim=extrema(dates))
        savefig(p, path*title*"_"*string(today())*"."*format)
    end

    function line_plot(data::SciMLBase.ODESolution, timeperiod, path="", title = "title", format="png")
        # TODO: capire perche' hanno due dimensioni diverse
        data = select!(DataFrame(data), Not(:timestamp))
        line_plot(data[1:nrow(data)-1,:], timeperiod, path, title, format)
    end
end