module pplot
    using Plots, LaTeXStrings, StatsPlots
    using InteractiveDynamics, CairoMakie
    using DataFrames, SciMLBase, Dates

    function static_preplot!(ax, model)
        CairoMakie.hidedecorations!(ax)

        # mostro posizione attrattore
        obj = CairoMakie.scatter!([model.attr_pos[1] model.attr_pos[2]]; color = :blue)
        CairoMakie.translate!(obj, 0, 0, 5)
    end

    # different epidemic states: S, E, I, R
    colors(a) = a.status == :S ? "grey80" : a.status == :E ? "yellow" : a.status == :I ? "red" : "green"

    function record_video(model, astep, mstep;
        name = "img/sngraph_"*string(today())*".mp4", framerate = 15, frames = 100, 
        title = "title")
        abmvideo(
            name, model, astep, mstep;
            framerate=framerate, frames=frames, 
            title=title, static_preplot!, ac = colors,
        )
    end

    function line_plot(sol, title = "title")
		p = Plots.plot(sol, labels = [L"S" L"E" L"I" L"R" L"D"], title = title, lw = 2, xlabel = L"Days")
        savefig(p, "img/"*title*"_"*string(today())*".png")
	end

	# function area_plot(sol, title = "title")
	# 	p = areaplot(sol.t, sol', labels = [L"S" L"E" L"I" L"R" L"D"], title = title, xlabel = L"Days")
    #     savefig(p, "img/"*title*"_"*string(today())*".png")
	# end

    # function line_plot(data::DataFrame, title = "title")
    #     p = @df data Plots.plot(cols(), title = title, lw = 2, xlabel = L"Days")
    #     savefig(p, "img/"*title*"_"*string(today())*".png")
    # end

    # function line_plot(data::Vector{Vector{Float64}}, title = "title")
    #     p = Plots.plot(data, labels = [L"S" L"E" L"I" L"R" L"D"], title = title, lw = 2, xlabel = L"Days")
    #     savefig(p, "img/"*title*"_"*string(today())*".png")
    # end

    # function area_plot(data::Vector{Vector{Float64}}, title = "title")
    #     p = areaplot(data.T, data', labels = [L"S" L"E" L"I" L"R" L"D"], title = title, lw = 2, xlabel = L"Days")
    #     savefig(p, "img/"*title*"_"*string(today())*".png")
    # end

end