module pplot
    using Plots, LaTeXStrings, StatsPlots
    using InteractiveDynamics, CairoMakie
    using DataFrames, SciMLBase, Dates, CSV
    using Agents, GraphMakie
    using Statistics: mean

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

    # create problems for reasons
    graphplotkwargs = (
        layout = GraphMakie.Shell(),
        arrow_show = false,
        edge_color = edge_color,
        edge_width = edge_width,
        edge_plottype = :linesegments
    )

    function get_adata()
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
		quarantined(x) = count(i == :Q for i in x)
        recovered(x) = count(i == :R for i in x)
        happiness(x) = mean(x)

        return [(:status, susceptible), (:status, exposed), (:status, infected), (:status, quarantined), (:status, recovered), (:happiness, happiness)]
    end

    function get_mdata(model)
        dead(model) = sum(model.number_point_of_interest) - nagents(model)
        return [dead]
    end

    function custom_layout(fig, abmobs, step, name)
        plot_layout = fig[:, end+1] = GridLayout()
        count_layout = plot_layout[1, 1] = GridLayout()
        s = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).susceptible_status))
        e = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).exposed_status))
        i = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).infected_status))
        q = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).quarantined_status))
        r = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).recovered_status))
        d = @lift(Point2f.($(abmobs.mdf).step, $(abmobs.mdf).dead))

        happiness = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).happiness_happiness))

        ax_seir = Axis(count_layout[1, 1]; ylabel="SEIR Dynamic")
        scatterlines!(ax_seir, s; label="susceptible")
        scatterlines!(ax_seir, e; label="exposed")
        scatterlines!(ax_seir, i; label="infected")
        scatterlines!(ax_seir, q; label="quarantined")
        scatterlines!(ax_seir, r; label="recovered")
        scatterlines!(ax_seir, d; label="dead")

        Legend(count_layout[1, 2], ax_seir;)
        
        # TODO: cambiare tipologia di plot?
        ax_happiness = Axis(count_layout[2, 1]; ylabel="Cumulative happiness")
        scatterlines!(ax_happiness, happiness; label="happiness")
        # non si noterebbe una linea che rimane tra [-1,1] altrimenti
        # scatterlines!(ax_happiness, q; label="quarantined")
        # scatterlines!(ax_happiness, d; label="dead")

        Legend(count_layout[2, 2], ax_happiness;)

        on(abmobs.model) do m
            autolimits!(ax_happiness)
            autolimits!(ax_seir)
        end

        record(fig, name) do io
            for _ in 1:step
                recordframe!(io)
                Agents.step!(abmobs, 1)
            end
            recordframe!(io)
        end
    end

    # FIXME: not plot covid evolution, but why?
    function custom_video(model, astep, mstep; 
        title="title", path="img/", format=".mkv", 
        framerate = 15, frames = 100)
        isdir(path) == false && mkpath(path)
        name = path*title*"_"*string(today())*format

        # experiment custom plot
        fig, ax, abmobs = abmplot(model;
        agent_step! = astep, model_step! = mstep, 
        as=city_size, ac=city_color, graphplotkwargs...,
        adata=get_adata(), mdata=get_mdata(model), figure=(; resolution=(1600,800)))
        custom_layout(fig, abmobs, frames, name)
    end

    function video(model, astep, mstep; 
        title="title", path="img/", format=".mkv", 
        framerate = 15, frames = 100)
        isdir(path) == false && mkpath(path)
        name = path*title*"_"*string(today())*format

        abmvideo(name, model, astep, mstep;
            framerate=framerate, frames=frames,
            title=title, as=city_size, ac=city_color, graphplotkwargs...)
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