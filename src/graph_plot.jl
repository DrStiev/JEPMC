# modulo per il plot dei grafici
module graph_plot
    using GraphMakie
    using InteractiveDynamics
	using GLMakie
    using Agents

    include("graph_model.jl")

    function hist(model)
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
        return fig
    end

    function rec_animation(model, figure, step)
        abmobs = ABMObservable(model; graph_model.agent_step!)
        record(figure, "covid_evolution.mp4"; framerate = 5) do io
            for _ in 1:step
                recordframe!(io)
                Agents.step!(abmobs, 1)
            end
            recordframe!(io)
        end
    end

    # function model_status()
    #     susceptible(x) = count(i == :S for i in x)
    #     infected(x) = count(i == :I for i in x)
    #     recovered(x) = count(i == :R for i in x)
    #     adata = [(:status, f) for f in (susceptible, infected, recovered)]
    #     dead(model) = sum(model.Ns) - nagents(model)
    #     mdata = [dead]
    #     return adata, mdata
    # end

    # function plot(model)
    #     adata, mdata = model_status()
    #     # https://juliadynamics.github.io/Agents.jl/stable/agents_visualizations/#GraphSpace-models-1
    #     city_size(agent) = 0.005 * length(agent)
    #     function city_color(agent)
    #         agent_size = length(agent)
    #         infected = count(a.status == :I for a in agent)
    #         recovered = count(a.status == :R for a in agent)
    #         return RGBf(infected / agent_size, recovered / agent_size, 0)
    #     end

    #     edge_color(model) = fill((:grey, 0.25), Agents.Graphs.ne(model.space.graph))
    #     function edge_width(model)
    #         w = zeros(Agents.Graphs.ne(model.space.graph))
    #         for e in Agents.Graphs.edges(model.space.graph)
    #             push!(w, 0.004 * length(model.space.stored_ids[e.src]))
    #             push!(w, 0.004 * length(model.space.stored_ids[e.dst]))
    #         end
    #         filter!(>(0), w)
    #         return w
    #     end

    #     graphplotkwargs = (
    #         layout = GraphMakie.Shell(), # posizione nodi
    #         arrow_show = false, # mostrare archi orientati
    #         edge_color = edge_color,
    #         edge_width = edge_width,
    #         edge_plottype = :linesegments # needed for tapered edge widths
    #     )

    #     fig, ax, abmobs = abmplot(model;
    #         agent_step! = agent_step!, 
    #         model_step! = dummystep, # model_step!,
    #         # params,
    #         as = city_size, 
    #         ac = city_color, 
    #         graphplotkwargs,
    #         adata,
    #         mdata,
    #     )
    #     # creo un nuovo plot layout
    #     plot_layout = fig[:, end + 1] = GridLayout()
    #     # creo un sublayout
    #     count_layout = plot_layout[1, 1] = GridLayout()

    #     infected = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).infected_status))
    #     susceptible = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).susceptible_status))
    #     recovered = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).recovered_status))
    #     dead = @lift(Point2f.($(abmobs.mdf).dead))

    #     ax_s = Axis(count_layout[1, 1]; ylabel = "Susceptible", xlabel = "Giorni")
    #     scatterlines!(ax_s, susceptible; color = "grey80", label = "Susceptible")
    #     ax_i = Axis(count_layout[2, 1]; ylabel = "Infected", xlabel = "Giorni")
    #     scatterlines!(ax_i, infected; color = "red2", label = "Infected")
    #     ax_r = Axis(count_layout[3, 1]; ylabel = "Recovered", xlabel = "Giorni")
    #     scatterlines!(ax_r, recovered; color = "green", label = "Recovered")
    #     ax_d = Axis(count_layout[4, 1]; ylabel = "Dead", xlabel = "Giorni")
    #     scatterlines!(ax_d, dead; color = :black, label = "Dead")

    #     # ad ogni aggiornamento dell'observable aggiusto gli assi
    #     on(abmobs.model) do m
    #         autolimits!(ax_s)
    #         autolimits!(ax_i)
    #         autolimits!(ax_r)
    #         autolimits!(ax_d)
    #     end
    #     return fig, abmobs
    # end
end