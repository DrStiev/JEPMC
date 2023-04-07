module pplot
    using Plots, LaTeXStrings, StatsPlots
    using InteractiveDynamics, CairoMakie

    function static_preplot!(ax, model)
        obj = CairoMakie.scatter!([50,50]; color = :blue)
        CairoMakie.hidedecorations!(ax)
        CairoMakie.translate!(obj, 0, 0, 5)
    end

    function record_video(model, astep, mstep;
        name = "social_network_graph.mp4", framerate = 15, frames = 100, 
        title = "social network graph model", preplot = static_preplot!)
        abmvideo(
            name, model, astep, mstep;
            framerate=framerate, frames=frames, 
            title=title, preplot,
        )
    end

    function line_plot(sol{<: Vector{Vectro{Float64}}}, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered" L"Dead"], title = "SEIRD Dynamics")
		return Plots.plot(sol, labels = labels, title = title, lw = 2, xlabel = L"Days")
	end

	function area_plot(sol{<: Vector{Vectro{Float64}}}, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered" L"Dead"], title = "SEIRD Dynamics")
		return areaplot(sol.t, sol', labels = labels, title = title, xlabel = L"Days")
	end

    function collect(model, astep, mstep; n = 1000)
        susceptible(x) = count(i == :S for i in x)
        exposed(x) = count(i == :E for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)
        dead(x) = model.N - nagents(model)

        to_collect = [(:status, f) for f in (susceptible, exposed, infected, recovered, dead)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
        return data
    end

    function line_plot(data{<: DataFrame}, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered" L"Dead"], title = "ABM GraphSpace Dynamics")
        @df data = [data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]]
        return Plots.plot(data, labels = labels, title = title, lw = 2, xlabel = L"Days")
    end

    function area_plot(data{<: DataFrame}, labels = [L"Susceptible" L"Exposed" L"Infected" L"Recovered" L"Dead"], title = "ABM GraphSpace Dynamics")
        @df data = [data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]]
        return areaplot(data.t, data', labels = labels, title = title, lw = 2, xlabel = L"Days")
end