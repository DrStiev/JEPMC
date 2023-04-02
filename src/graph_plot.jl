# modulo per il plot dei grafici
module graph_plot
    using GraphMakie
    using InteractiveDynamics
	using GLMakie
    using Agents

    function line_plot(model, step, n)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        quarantined(x) = count(i == :Q for i in x)

        to_collect = [(:status, f) for f in (exposed, infected, recovered, quarantined, length)]
        data, _ = run!(model, step, n; adata = to_collect)

        N = sum(model.Ns)
        x = data.step
        fig = Figure(resolution = (600, 400))
        ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
        le = scatterlines!(ax, x, log10.(data[:, aggname(:status, exposed)]), color = "aquamarine2")
        li = scatterlines!(ax, x, log10.(data[:, aggname(:status, infected)]), color = "red2")
        lr = scatterlines!(ax, x, log10.(data[:, aggname(:status, recovered)]), color = "green")
        lq = scatterlines!(ax, x, log10.(data[:, aggname(:status, quarantined)]), color = "burlywood4")
        dead = log10.(N .- data[:, aggname(:status, length)])
        ld = scatterlines!(ax, x, dead, color = "black")
        Legend(fig[1, 2], [le, li, lr, lq, ld], ["Exposed", "Infected", "Recovered", "Quarantined", "Dead"])
        return fig, data
    end
end