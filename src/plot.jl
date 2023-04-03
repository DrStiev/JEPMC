# modulo per il plot dei grafici
module graph_plot
    using GraphMakie
    using InteractiveDynamics
	using GLMakie
    using Agents
    using Plots

    # FIXME: display sbagliato
    function line_plot(N, data)
        x = data.step
        fig = Figure(resolution = (600, 400))
        ax = fig[1, 1] = Axis(fig, xlabel = "steps", ylabel = "log10(count)")
        ls = lines!(ax, x, log10.(data[:, aggname(:susceptible_status)]), color = "grey80")
        le = lines!(ax, x, log10.(data[:, aggname(:exposed_status)]), color = "aquamarine2")
        li = lines!(ax, x, log10.(data[:, aggname(:infected_status)]), color = "red2")
        lr = lines!(ax, x, log10.(data[:, aggname(:recovered_status)]), color = "green")
        lq = lines!(ax, x, log10.(data[:, aggname(:quarantined_status)]), color = "burlywood4")
        dead = log10.(N .- data[:, aggname(:length_status)])
        ld = lines!(ax, x, dead, color = "black")
        Legend(fig[1, 2], [ls, le, li, lr, lq, ld], ["Susceptible", "Exposed", "Infected", "Recovered", "Quarantined", "Dead"])
        return fig
    end     
end