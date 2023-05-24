module pplot
using Plots, LaTeXStrings, StatsPlots
using InteractiveDynamics, CairoMakie
using DataFrames, Dates, CSV
using Agents, GraphMakie, GLMakie
using Statistics: mean
using ProgressMeter

GLMakie.activate!()

city_size(agents) = 0.005 * length(agents)
function city_color(agents)
    agent = length(agents)
    exposed = count(a.status == :E for a in agents)
    infected = count(a.status == :I for a in agents)
    recovered = count(a.status == :R for a in agents)
    # vaccined = count(a.detected == :V for a in agents)
    return RGBf(infected / agent, recovered / agent, 0)
end

edge_color(model) = fill((:grey, 0.25), GraphMakie.Graphs.ne(model.space.graph))
function edge_width(model)
    w = zeros(GraphMakie.Graphs.ne(model.space.graph))
    for e in GraphMakie.Graphs.edges(model.space.graph)
        push!(w, 0.004 * length(model.space.stored_ids[e.src]))
        push!(w, 0.004 * length(model.space.stored_ids[e.dst]))
    end
    filter!(>(0), w)
    return w
end

graphplotkwargs = (
    layout=GraphMakie.Spring(),
    arrow_show=true,
    arrow_shift=:end,
    edge_color=edge_color,
    edge_width=edge_width,
    edge_plottype=:linesegments,
)

function get_adata()
    # information about the model
    susceptible(x) = count(i == :S for i in x)
    exposed(x) = count(i == :E for i in x)
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)
    happiness(x) = mean(x)

    return [
        (:status, susceptible),
        (:status, exposed),
        (:status, infected),
        (:status, recovered),
        (:happiness, happiness),
    ]
end

function get_mdata(model)
    dead(model) = sum(model.number_point_of_interest) - nagents(model)
    R₀(model) = model.R₀
    η(model) = model.η
    return [dead, R₀, η]
end

function custom_layout(fig, abmobs, step, name)
    plot_layout = fig[:, end+1] = GridLayout()
    count_layout = plot_layout[1, 1] = GridLayout()
    # get information about the general epidemic
    s = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).susceptible_status))
    e = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).exposed_status))
    i = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).infected_status))
    r = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).recovered_status))
    d = @lift(Point2f.($(abmobs.mdf).step, $(abmobs.mdf).dead))

    # get information about the general mood of the society
    happiness = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).happiness_happiness))

    # get information about the R₀ index
    Rₜ = @lift(Point2f.($(abmobs.mdf).step, $(abmobs.mdf).R₀))

    # get information about the countermeasures applied
    η = @lift(Point2f.($(abmobs.mdf).step, $(abmobs.mdf).η))

    ax_seir = Axis(count_layout[1, 1]; ylabel="SEIR Dynamic")
    lines!(ax_seir, s; label="susceptible")
    lines!(ax_seir, e; label="exposed")
    lines!(ax_seir, i; label="infected")
    lines!(ax_seir, r; label="recovered")
    lines!(ax_seir, d; label="dead")
    Legend(count_layout[1, 2], ax_seir;)

    ax_happiness = Axis(count_layout[2, 1]; ylabel="Average happiness")
    lines!(ax_happiness, happiness; label="happiness")
    Legend(count_layout[2, 2], ax_happiness;)

    ax_R₀ = Axis(count_layout[3, 1]; ylabel="Reproduction number")
    lines!(ax_R₀, Rₜ; label="R₀")
    Legend(count_layout[3, 2], ax_R₀;)

    ax_eta = Axis(count_layout[4, 1]; ylabel="Countermeasures strickness")
    lines!(ax_eta, η; label="η")
    Legend(count_layout[4, 2], ax_eta;)

    p = if typeof(step) <: Int
        ProgressMeter.Progress(step; enabled=true, desc="run! progress: ")
    else
        ProgressMeter.ProgressUnknown(desc="run! steps done: ", enabled=showprogress)
    end

    on(abmobs.model) do m
        autolimits!(ax_happiness)
        autolimits!(ax_eta)
        autolimits!(ax_seir)
        autolimits!(ax_R₀)

        ProgressMeter.next!(p)
    end

    # GLMakie do not work on kos 
    record(fig, name) do io
        for _ = 1:step
            recordframe!(io)
            Agents.step!(abmobs, 1)
        end
        recordframe!(io)
    end
end

function custom_video(
    model,
    astep,
    mstep;
    title="title",
    path="img/",
    format=".mkv",
    frames=100
)
    isdir(path) == false && mkpath(path)
    name = path * title * "_" * string(today()) * format

    fig, ax, abmobs = abmplot(
        model;
        (agent_step!)=astep,
        (model_step!)=mstep,
        as=city_size,
        ac=city_color,
        graphplotkwargs=graphplotkwargs,
        adata=get_adata(),
        mdata=get_mdata(model),
        figure=(; resolution=(1600, 800))
    )
    custom_layout(fig, abmobs, frames, name)
end

function save_plot(plot, path="", title="title", format="png")
    isdir(path) == false && mkpath(path)
    savefig(plot, path * title * "_" * string(today()) * "." * format)
end

function loss_plot(losses, path="", title="title", format="png")
    isdir(path) == false && mkpath(path)
    p = Plots.plot(
        losses,
        yaxis=:log,
        xaxis=:log,
        xlabel="Iterations",
        ylabel="loss",
    )
    savefig(p, path * title * "_" * string(today()) * "." * format)
end
end
