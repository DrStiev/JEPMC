module test_parameters
	include("params.jl")

	@time df = model_params.get_data("data/italy/")
	
	@time ode_parameters = model_params.extract_params(df)
	@show ode_parameters
	@time abm_parameters = model_params.extract_params(df, 8, (50,5000), 0.01)
	@show abm_parameters
end

module test_plot
	using DataFrames
	include("params.jl")
	include("pplot.jl")

	df = model_params.get_data("data/italy/")
	@time pplot.line_plot(select(df, [:nuovi_positivi]), 
		df[!, :data], "img/data_plot/", "nuovi_positivi", "png")
	@time pplot.line_plot(select(df, [:nuovi_positivi]), 
		df[!, :data], "img/data_plot/", "nuovi_positivi", "pdf")
	@time pplot.line_plot(select(df, [:totale_positivi, :totale_ospedalizzati, :dimessi_guariti, :deceduti]), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "png")
	@time pplot.line_plot(select(df, [:totale_positivi, :totale_ospedalizzati, :dimessi_guariti, :deceduti]), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "pdf")
end

# TODO: to be implemented
module test_uode
	using Plots
	include("uode.jl")
	include("params.jl")
	include("pplot.jl")

	# df = model_params.get_data("data/italy/")
	# u0, p, tspan = model_params.extract_params(df)

	# prob = ode.get_ODE_problem(ode.SEIR, u0, tspan, p)
	# @time sol = ode.get_solution(prob)

	# pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "png")
	# pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "pdf")
end

module test_abm
	using Agents, DataFrames
	include("params.jl")
	include("pplot.jl")
	include("graph.jl")

	df = model_params.get_data("data/italy/")
	abm_parameters = model_params.extract_params(df, 8, (50, 5000), 0.01)
	model = graph.init(; abm_parameters...)
	# TODO: understand why the general happiness tend to be always negative
	@time pplot.custom_video(model, graph.agent_step!, Agents.dummystep; title="graph_agent_custom", path="img/video/", format=".mp4", frames=365)
	model = graph.init(; abm_parameters...)
	@time pplot.custom_video(model, graph.agent_step!, Agents.dummystep; title="graph_agent_custom", path="img/video/", format=".mkv", frames=365)

	model = graph.init(; abm_parameters...)
	@time data = graph.collect(model, graph.agent_step!; n=365)
	pplot.line_plot(select(data, Not(:happiness_happiness)), df[1:length(data[!,1]),:data], "img/abm/", "graph_agent", "png")
	pplot.line_plot(select(data, Not(:happiness_happiness)), df[1:length(data[!,1]),:data], "img/abm/", "graph_agent", "pdf")
	pplot.line_plot(select(data, [:happiness_happiness]), df[1:length(data[!,1]),:data], "img/abm/", "graph_agent_happiness", "png")
	pplot.line_plot(select(data, [:happiness_happiness]), df[1:length(data[!,1]),:data], "img/abm/", "graph_agent_happiness", "pdf")
end

module test_controller
	include("params.jl")
	include("pplot.jl")
	include("controller.jl")
end