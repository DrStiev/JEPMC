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

    # TODO: add different plots
    function line_plot(data::DataFrame, title = "title")
        p = @df data Plots.plot(cols(), title = title, lw = 2, xlabel = L"Days")
        savefig(p, "img/"*title*"_"*string(today())*".png")
    end

    function line_plot(data::SciMLBase.EnsembleSummary, title = "title")
        p = Plots.plot(data, idxs = (2,), labels = [L"susceptible" L"exposed" L"infected" L"recovered" L"dead" L"R0" L"mortality"], title = title, lw = 2, xlabel = L"Days", legend = :topright)
        savefig(p, "img/"*title*"_"*string(today())*".png")
    end

    function line_plot(data, model, title="title")
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)

        N = sum(model.number_point_of_interest)
        x = data.step
        fig = Figure()
        ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
        ls = lines!(ax, x, log10.(data[:, aggname(:status, susceptible)]))
        le = lines!(ax, x, log10.(data[:, aggname(:status, exposed)]))
        li = lines!(ax, x, log10.(data[:, aggname(:status, infected)]))
        lr = lines!(ax, x, log10.(data[:, aggname(:status, recovered)]))
        dead = log10.(N .- data[:, aggname(:status, length)])
        ld = lines!(ax, x, dead)
        Legend(fig[1, 2], [ls, le, li, lr, ld], ["susceptible", "exposed", "infected", "recovered", "dead"])
        Makie.save("img/"*title*"_"*string(today())*".png", fig)
    end

    function save_parameters(model, title = title)
        df = DataFrame(model.properties)
        CSV.write("data/"*title*"_"*string(today()), df)
    end
end