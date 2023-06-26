using Agents, DataFrames, Plots, Distributions, Random, Dates, Distributed
using Statistics: mean

@everywhere include("utils.jl")
@everywhere include("controller.jl")
@everywhere include("graph.jl")
@everywhere include("ode.jl")

gr()

function save_plot(plot, path::String="", title::String="title", format::String="png")
    isdir(path) == false && mkpath(path)
    savefig(plot, path * title * "_" * string(today()) * "." * format)
end

function split_dataset(data::DataFrame)
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
    i = 1
    maxiter = 100

    p = parameters.get_abm_parameters(20, 0.01, 3300)

    while i ≤ maxiter
        try
            # se non runno ad ogni try il modello, se fallisce una volta allora
            # fallira' sempre. Ogni try e' associato ad un nuovo modello.
            model = graph.init(; p...)
            data = graph.collect(model; n=250, showprogress=true)
            d, _, _ = split_dataset(data)

            eq, (prob, sol) = SysId.system_identification(d; saveplot=true, verbose=true)
            p = plot(
                plot(prob, title="SYSTEM IDENTIFICATION AFTER $i ITERATIONS"),
                plot(sol),
            )
            save_plot(p, "img/system_identification/", "SYSTEM IDENTIFICATION", "pdf")
            println("iteration $i of $maxiter successful")
            break
        catch ex
            println("iteration $i of $maxiter failed because of $ex")
            i += 1
        end
    end
end

test_system_identification()

function test_prediction()
    i = 1
    maxiter = 100

    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    n = 250
    sp = n * 5 # prediction

    while i ≤ maxiter
        try
            model = graph.init(; abm_parameters...)
            data = graph.collect(model; n=sp, showprogress=true)
            ddata, _, _ = split_dataset(data)
            Xₙ = Array(ddata)
            p_true = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0]

            pred = udePredict.ude_prediction(
                ddata[1:n, :],
                p_true,
                round(Int, sp / 2);
                lossTitle="LOSS",
                plotLoss=true,
                maxiters=2500
            )
            p = scatter(
                Array(Xₙ ./ sum(Xₙ[1, :])),
                label=["True S" "True E" "True I" "True R" "True D"],
            )
            plot!(
                transpose(pred[1]),
                xlabel="t",
                label=["Estimated S" "Estimated E" "Estimated I" "Estimated R" "Estimated D"],
                title="NN Approximation",
            )
            plot!(
                p,
                [n - 0.01, n + 0.01],
                [0.0, 1.0],
                lw=2,
                color=:black,
                label=nothing,
            )
            annotate!([(n / 3, 1.0, text("Training Data", :center, :top, :black))])
            # test symbolic regression
            long_time_estimation = udePredict.symbolic_regression(
                pred[1],
                pred[2],
                p_true,
                sp;
                maxiters=10_000
            )

            p1 = scatter(
                Array(Xₙ ./ sum(Xₙ[1, :])),
                label=["True S" "True E" "True I" "True R" "True D"],
            )
            plot!(
                long_time_estimation,
                xlabel="t",
                label=["Estimated S" "Estimated E" "Estimated I" "Estimated R" "Estimated D"],
                title="NN + SINDy Approximation",
            )
            plot!(
                p1,
                [n - 0.01, n + 0.01],
                [0.0, 1.0],
                lw=2,
                color=:black,
                label=nothing,
            )
            annotate!([(n / 3, 1.0, text("Training Data", :center, :top, :black))])
            l = @layout [
                grid(1, 1)
                grid(1, 1)
            ]
            pt = plot(plot(p), plot(p1), layout=l)
            save_plot(pt, "img/prediction/", "PREDICTION", "pdf")
            println("iteration $i of $maxiter successful")
            break
        catch ex
            # try to free as much memory as possible after each try
            GC.gc()
            println("iteration $i of $maxiter failed because of $ex")
            i += 1
        end
    end
end

test_prediction()

function test_abm()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)

    data = graph.collect(model; n=1200, showprogress=true)
    graph.save_dataframe(data, "data/abm/", "ABM SEIR NO INTERVENTION")

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

function test_abm_with_controller()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300; controller=true)
    model = graph.init(; abm_parameters...)

    data = graph.collect(model; n=1200, showprogress=true)
    graph.save_dataframe(data, "data/abm/", "ABM SEIR WITH CONTROLLER AND INTERVENTION")

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
    save_plot(p, "img/abm/", "ABM SEIR WITH CONTROLLER AND INTERVENTION", "pdf")
end


function test_ensemble()
    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    models = [graph.init(; seed=i, abm_parameters...) for i in rand(UInt64, 100)]
    data = graph.ensemble_collect(models; n=1200, showprogress=true, parallel=true)
    graph.save_dataframe(data, "data/ensemble/", "ENSEMBLE ABM SEIR NO INTERVENTION")
    ens_data = dataset.read_dataset(
        "data/ensemble/ENSEMBLE ABM SEIR NO INTERVENTION_" * string(today()) * ".csv",
    )
    d = [filter(:ensemble => ==(i), ens_data) for i in unique(ens_data[!, :ensemble])]
    res_seir = DataFrame(
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [
            "lb_s",
            "avg_s",
            "ub_s",
            "lb_e",
            "avg_e",
            "ub_e",
            "lb_i",
            "avg_i",
            "ub_i",
            "lb_r",
            "avg_r",
            "ub_r",
            "lb_d",
            "avg_d",
            "ub_d",
        ],
    )
    res_hη = DataFrame(
        [[], [], [], [], [], []],
        ["lb_h", "avg_h", "ub_h", "lb_η", "avg_η", "ub_η"],
    )
    res_r0 = DataFrame([[], [], []], ["lb_R0", "avg_R0", "ub_R0"])
    for i = 1:length(d[1][:, 1])
        s = [d[j][i, :susceptible_status] for j = 1:length(d)]
        e = [d[j][i, :exposed_status] for j = 1:length(d)]
        is = [d[j][i, :infected_status] for j = 1:length(d)]
        r = [d[j][i, :recovered_status] for j = 1:length(d)]
        dd = [d[j][i, :dead] for j = 1:length(d)]

        ms = mean(s)
        me = mean(e)
        mi = mean(is)
        mr = mean(r)
        md = mean(dd)
        push!(
            res_seir,
            [
                ms - minimum(s),
                ms,
                maximum(s) - ms,
                me - minimum(e),
                me,
                maximum(e) - me,
                mi - minimum(is),
                mi,
                maximum(is) - mi,
                mr - minimum(r),
                mr,
                maximum(r) - mr,
                md - minimum(dd),
                md,
                maximum(dd) - md,
            ],
        )

        h = [d[j][i, :happiness] for j = 1:length(d)]
        η = [d[j][i, :active_countermeasures] for j = 1:length(d)]
        mh = mean(h)
        mη = mean(η)
        push!(
            res_hη,
            [
                mh - minimum(h),
                mean(h),
                maximum(h) - mh,
                mη - minimum(η),
                mean(η),
                maximum(η) - mη,
            ],
        )

        r0 = [d[j][i, :R₀] for j = 1:length(d)]
        mr0 = mean(r0)
        push!(res_r0, [mr0 - minimum(r0), mean(r0), maximum(r0) - mr0])
    end

    p1 = plot(
        res_seir[!, :avg_s],
        ribbon=(res_seir[!, :lb_s], res_seir[!, :ub_s]),
        labels="Susceptible",
        title="ABM Dynamics",
    )
    plot!(
        res_seir[!, :avg_e],
        ribbon=(res_seir[!, :lb_e], res_seir[!, :ub_e]),
        labels="Exposed",
    )
    plot!(
        res_seir[!, :avg_i],
        ribbon=(res_seir[!, :lb_i], res_seir[!, :ub_i]),
        labels="Infected",
    )
    plot!(
        res_seir[!, :avg_r],
        ribbon=(res_seir[!, :lb_r], res_seir[!, :ub_r]),
        labels="Recovered",
    )
    plot!(
        res_seir[!, :avg_d],
        ribbon=(res_seir[!, :lb_d], res_seir[!, :ub_d]),
        labels="Dead",
    )

    p2 = plot(
        res_hη[!, :avg_h],
        ribbon=(res_hη[!, :lb_h], res_hη[!, :ub_h]),
        labels="happiness",
        title="Agents response to η",
    )
    plot!(res_hη[!, :avg_η], ribbon=(res_hη[!, :lb_η], res_hη[!, :ub_η]), labels="η")

    p3 = plot(
        res_r0[!, :avg_R0],
        ribbon=(res_r0[!, :lb_R0], res_r0[!, :ub_R0]),
        labels="R₀",
        title="Reproduction number",
    )

    l = @layout [
        grid(1, 1)
        grid(1, 2)
    ]
    p = plot(p1, p2, p3, layout=l)
    save_plot(p, "img/ensemble/", "ENSEMBLE SEIR NO INTERVENTION", "pdf")
end

test_ensemble()

function test_ode()
    # must be between [0-1] for numerical stability
    u, p, t = parameters.get_ode_parameters(20, 3300)
    prob = ode.get_ode_problem(ode.seir!, u, t, p)
    sol = ode.get_ode_solution(prob)

    p = plot(
        sol,
        labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
        title="SEIR Dynamics NO INTERVENTION",
    )
    save_plot(p, "img/ode/", "ODE SEIR NO INTERVENTION", "pdf")
end

test_ode()

function test_comparison_abm_ode()
    u, p, t = parameters.get_ode_parameters(20, 3300)
    prob = ode.get_ode_problem(ode.seir!, u, t, p)
    sol = ode.get_ode_solution(prob)
    p1 = plot(
        sol,
        labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
        title="SEIR Dynamics R₀ = $(p[1])",
    )

    abm_parameters = parameters.get_abm_parameters(20, 0.01, 3300)
    model = graph.init(; abm_parameters...)
    data = graph.collect(model; n=1200, showprogress=true)
    x, _, _ = split_dataset(data)
    x = x ./ sum(x[1, :])
    l = @layout [
        grid(1, 1)
        grid(1, 1)
    ]
    p2 = plot(
        Array(x),
        labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
        title="ABM Dynamics R₀ = $(abm_parameters[:R₀])",
    )
    p = plot(plot(p2), plot(p1), layout=l)
    save_plot(p, "img/abm/", "ABM-ODE COMPARISON", "pdf")
end

test_comparison_abm_ode()
