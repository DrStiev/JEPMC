using Agents, DataFrames, Plots, Distributions, Random, Dates
using Statistics: mean

using Distributed

include("utils.jl")
include("graph.jl")
include("controller.jl")
include("uode.jl")

function save_plot(plot, path = "", title = "title", format = "png")
    isdir(path) == false && mkpath(path)
    savefig(plot, path * title * "_" * string(today()) * "." * format)
end

function split_dataset(data)
    p1 = select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    )
    p2 = select(data, [:active_countermeasures, :happiness])#_happiness])
    p3 = select(data, [:R₀])
    return p1, p2, p3
end

function test_dataset(url::String, path::String, iso_code::String)
    dataset.download_dataset(path, url)
    df = dataset.read_dataset("data/OWID/owid-covid-data.csv")
    return dataset.dataset_from_location(df, iso_code)
end

test_dataset(
    "https://covid.ourworldindata.org/data/owid-covid-data.csv",
    "data/OWID/",
    "ITA",
)

function test_parameters()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    ode_parameters = parameters.get_ode_parameters(20, 3300)
    parameters.save_parameters(abm_parameters, "data/parameters/", "abm_parameters")
    parameters.load_parameters("data/parameters/abm_parameters.jld2")
end

test_parameters()

function plot_current_situation(path::String, iso_code::String)
    date, day_info, total_count, R₀ =
        dataset.dataset_from_location(dataset.read_dataset(path), iso_code)
    p = plot(
        plot(
            Array(day_info),
            labels = ["Infected" "Tests" "Vaccinations" "Deaths"],
            title = "Detected Dynamics",
        ),
        plot(
            Array(total_count),
            labels = ["Susceptible" "Infected" "Deaths" "Tests"],
            title = "Overall Dynamics",
        ),
        plot(Array(R₀), labels = "R₀", title = "Reproduction Rate"),
    )
    save_plot(p, "img/data_plot/", "cumulative_plot", "pdf")
end

plot_current_situation("data/OWID/owid-covid-data.csv", "ITA")

function test_system_identification()
    p = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; p...)
    data = graph.collect(model; n = 30, showprogress = true)

    d = select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    )

    eq = SysId.system_identification(d)
    println("equation: $eq")
end

test_system_identification()

function test_abm()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)

    data = graph.collect(model; n = 1200, showprogress = true)
    graph.save_dataframe(data, "data/abm/", "ABM SEIR NO INTERVENTION")
    df = graph.load_dataset("data/abm/ABM SEIR NO INTERVENTION_" * string(today()) * ".csv")

    p1, p2, p3 = split_dataset(data)
    l = @layout [
        grid(1, 1)
        grid(1, 2)
    ]
    p = plot(
        plot(
            Array(p1),
            labels = ["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
            title = "ABM Dynamics",
        ),
        plot(Array(p2), labels = ["η" "Happiness"], title = "Agents response to η"),
        plot(Array(p3), labels = "R₀", title = "Reproduction number"),
        layout = l,
    )
    save_plot(p, "img/abm/", "ABM SEIR NO INTERVENTION", "pdf")
end

test_abm()

function test_uode()
    # must be between [0-1] otherwise strange behaviour
    u, p, t = parameters.get_ode_parameters(20, 3300)
    prob = uode.get_ode_problem(uode.seir!, u, (1.0, 30), p)
    sol = uode.get_ode_solution(prob)

    p = plot(
        sol,
        labels = ["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
        title = "SEIR Dynamics NO INTERVENTION",
    )
    save_plot(p, "img/ode/", "ODE SEIR NO INTERVENTION", "pdf")
end

test_uode()

function test_controller()
    # https://link.springer.com/article/10.1007/s40313-023-00993-8
    p = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; p...)
    data = DataFrame()

    callback = function (plt)
        display(plt)
    end

    timeshift = 7
    for i = 1:trunc(Int, 100 / timeshift)
        dataₜ = graph.collect(model; n = timeshift - 1)
        data = vcat(data, dataₜ)
        controller.countermeasures!(model, dataₜ)
        p1, p2, p3, p4, p = split_dataset(data)
        callback(p)
    end

    # p1, p2, p3, p4, p = split_dataset(data)
    # controller.predict(p1, (0.0, length(p1[!, 1])))
end

test_controller()
