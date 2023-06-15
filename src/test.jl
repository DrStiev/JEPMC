using Agents, DataFrames, Plots, Distributions, Random, Dates
using Statistics: mean

include("utils.jl")
include("graph.jl")
include("controller.jl")
include("ode.jl")

gr()

function save_plot(plot, path="", title="title", format="png")
    isdir(path) == false && mkpath(path)
    savefig(plot, path * title * "_" * string(today()) * "." * format)
end

function split_dataset(data)
    p1 = select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    )
    p2 = select(data, [:active_countermeasures, :happiness])
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
    l = @layout [
        grid(1, 1)
        grid(1, 2)
    ]
    p = plot(
        plot(
            Array(day_info),
            labels=["Infected" "Tests" "Vaccinations" "Deaths"],
            title="Detected Dynamics",
        ),
        plot(
            Array(total_count),
            labels=["Susceptible" "Infected" "Deaths" "Tests"],
            title="Overall Dynamics",
        ),
        plot(Array(R₀), labels="R₀", title="Reproduction Rate"),
        layout=l,
    )
    save_plot(p, "img/data_plot/", "cumulative_plot", "pdf")
end

plot_current_situation("data/OWID/owid-covid-data.csv", "ITA")

function test_system_identification()
    p = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; p...)
    data = graph.collect(model; n=30, showprogress=true)

    d = select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    )

    eq = SysId.system_identification(d)
    # parameters.save_parameters(eq, "data/parameters/", "system identification")
end

test_system_identification()

function test_prediction()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)
    n = 30
    sp = 45 # short term prediction

    data = graph.collect(model; n=n * 2, showprogress=true)
    ddata = select(
        data,
        [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
    )
    Xₙ = Array(ddata)
    # Eye control
    pred, ts = udePredict.ude_prediction(
        ddata[1:n, :],
        sp;
        lossTitle="LOSS",
        plotLoss=true,
        maxiters=1000
    )
    p = plot(
        ts,
        transpose(pred[1]),
        xlabel="t",
        ylabel="s(t), e(t), i(t), r(t), d(t)",
        color=:red,
        label=["UDE Approximation" nothing],
    )
    scatter!(
        1:1.0:(n*2)+1,
        Array(Xₙ ./ sum(Xₙ[1, :])),
        color=:blue,
        label=["Measurements" nothing],
    )
    plot!(p, [n - 0.01, n + 0.01], [0.0, 1.0], lw=2, color=:black, label=nothing)
    annotate!([(
        float(n),
        1.0,
        text("End of Training Data", 10, :center, :top, :black, "Helvetica"),
    )])
    save_plot(p, "img/prediction/", "NN SHORT TERM", "pdf")
    # println(pred)
    # parameters.save_parameters(pred, "data/parameters/", "prediction")

    # test symbolic regression
    long_time_estimation =
        udePredict.symbolic_regression(pred[1], pred[2], n * 2; maxiters=1000)
    println(long_time_estimation)
    plot(
        long_time_estimation,
        xlabel="t",
        ylabel="s(t), e(t), i(t), r(t), d(t)",
        color=:red,
        label=["UDE Approximation" nothing],
    )
    plot!(
        1:1.0:(n*2)+1,
        Array(Xₙ ./ sum(Xₙ[1, :])),
        color=:blue,
        label=["Measurements" nothing],
    )
    save_plot(p, "img/prediction/", "NN AND SYNDY SHORT TERM", "pdf")
end

test_prediction()

function test_abm()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)

    data = graph.collect(model; n=1200, showprogress=true)
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
            labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
            title="ABM Dynamics",
        ),
        plot(Array(p2), labels=["η" "Happiness"], title="Agents response to η"),
        plot(Array(p3), labels="R₀", title="Reproduction number"),
        layout=l,
    )
    save_plot(p, "img/abm/", "ABM SEIR NO INTERVENTION", "pdf")
end

test_abm()

function test_controller()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)

    data = graph.collect(
        model;
        n=1200,
        showprogress=true,
        tshift=14,
        initial_training_data=30,
        maxiters=1000
    )
    graph.save_dataframe(data, "data/abm/", "ABM SEIR WITH INTERVENTION")
    df = graph.load_dataset("data/abm/ABM SEIR WITH INTERVENTION_" * string(today()) * ".csv")

    p1, p2, p3 = split_dataset(data)
    l = @layout [
        grid(1, 1)
        grid(1, 2)
    ]
    p = plot(
        plot(
            Array(p1),
            labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
            title="ABM Dynamics",
        ),
        plot(Array(p2), labels=["η" "Happiness"], title="Agents response to η"),
        plot(Array(p3), labels="R₀", title="Reproduction number"),
        layout=l,
    )
    save_plot(p, "img/abm/", "ABM SEIR WITH INTERVENTION", "pdf")
end

test_controller()

function test_ode()
    # must be between [0-1] for numerical stability
    u, p, t = parameters.get_ode_parameters(20, 3300)
    prob = ode.get_ode_problem(uode.seir!, u, (1.0, 1200.0), p)
    sol = ode.get_ode_solution(prob)

    p = plot(
        sol,
        labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
        title="SEIR Dynamics NO INTERVENTION",
    )
    save_plot(p, "img/ode/", "ODE SEIR NO INTERVENTION", "pdf")
end

test_ode()
