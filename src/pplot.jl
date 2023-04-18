module pplot
    using Plots, LaTeXStrings, StatsPlots
    using InteractiveDynamics, CairoMakie
    using DataFrames, SciMLBase, Dates, CSV
    using Agents, GraphMakie

    city_size(agents) = 0.005 * length(agents)
    function city_color(agents)
        agent = length(agents)
        exposed = count(a.status == :E for a in agents)
        infected = count(a.status == :I for a in agents)
        quarantined = count(a.status == :Q for a in agents)
        recovered = count(a.status == :R for a in agents)
        return RGBf((infected + exposed)/agent, recovered/agent, quarantined/agent)
    end

    edge_color(model) = fill((:grey, 0.25), GraphMakie.Graphs.ne(model.space.graph))
    function edge_width(model)
        w = zeros(GraphMakie.Graphs.ne(model.space.graph))
        for e in GraphMakie.Graphs.edges(model.space.graph)
            push!(w, 0.004 * length(model.space.stored_ids[e.src]))
            push!(w, 0.004 * length(model.space.stored_ids[e.dst]))
        end
        return w
    end

    graphplotkwargs = (
        layout = GraphMakie.Shell(),
        arrow_show = false,
        edge_color = edge_color,
        edge_width = edge_width,
        edge_plottype = :linesegments
    )

    function video(model, astep, mstep; 
        title="title", path="img/", framerate = 15, frames = 100)
        isdir(path) == false && mkpath(path)
        name = "img/"*title*"_"*string(today())*".mp4"
        abmvideo(name, model, astep, mstep;
            framerate=framerate, frames=frames,
            title=title, as=city_size, ac=city_color, graphplotkwargs)
    end

    function line_plot(data, timeperiod, path="", title = "title", format="png")
        isdir(path) == false && mkpath(path)
        dates = range(timeperiod[1], timeperiod[end], step=Day(1))
	    tm_ticks = round.(dates, Month(1)) |> unique;
        p = Plots.plot(timeperiod, Matrix(data), labels=permutedims(names(data)), 
            title=title, xticks=(tm_ticks, Dates.format.(tm_ticks, "uu/yyyy")), 
            xrot=45, xminorticks=true, xlim=extrema(dates))
        savefig(p, path*title*"_"*string(today())*"."*format)
    end

    function line_plot(data::SciMLBase.ODESolution, timeperiod, path="", title = "title", format="png")
        data = select!(DataFrame(data), Not(:timestamp))
        line_plot(data[1:length(timeperiod),:], timeperiod, path, title, format)
    end
end