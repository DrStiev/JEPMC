# modulo per il plot dei grafici
module graph_plot
    using GraphMakie
    using InteractiveDynamics
	using GLMakie
    using Agents

    include("graph_model.jl")

    function hist_animation(model, step)
        abmobs = ABMObservable(model; graph_model.agent_step!)

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
        barplot!(ax, model.Ns; strokecolor = :black, strokewidth = 1, color)

        record(fig, "covid_evolution.mp4"; framerate = 5) do io
            for _ in 1:step
                recordframe!(io)
                Agents.step!(abmobs, 1)
            end
            recordframe!(io)
        end
    end

    function line_plot(model, step)
        # susceptible(x) = count(i == :S for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        quarantined(x) = count(i == :Q for i in x)
        vaccinated(x) = count(i == :V for i in x)

        to_collect = [(:status, f) for f in (infected, recovered, quarantined, vaccinated, length)]
        data, _ = run!(model, graph_model.agent_step!, step; adata = to_collect)

        N = sum(model.Ns)
        x = data.step
        fig = Figure(resolution = (600, 400))
        ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
        # ls = scatterlines!(ax, x, log10.(data[:, aggname(:status, susceptible)]), color = "grey80")
        li = scatterlines!(ax, x, log10.(data[:, aggname(:status, infected)]), color = "red2")
        lr = scatterlines!(ax, x, log10.(data[:, aggname(:status, recovered)]), color = "green")
        lq = scatterlines!(ax, x, log10.(data[:, aggname(:status, quarantined)]), color = "burlywood4")
        lv = scatterlines!(ax, x, log10.(data[:, aggname(:status, vaccinated)]), color = "blue3")
        dead = log10.(N .- data[:, aggname(:status, length)])
        ld = scatterlines!(ax, x, dead, color = "black")
        Legend(fig[1, 2], [li, lr, lq, lv, ld], ["Infected", "Recovered", "Quarantined", "Vaccinated", "Dead"])
        return fig, data
    end
end