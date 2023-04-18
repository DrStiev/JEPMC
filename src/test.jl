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
	@time pplot.line_plot(select(df, [:variazione_totale_positivi]), 
		df[!, :data], "img/data_plot/", "rapporto-positivi-guariti", "png")
	@time pplot.line_plot(select(df, [:variazione_totale_positivi]), 
		df[!, :data], "img/data_plot/", "rapporto-positivi-guariti", "pdf")
	@time pplot.line_plot(select(df, [:totale_positivi, :totale_ospedalizzati, :dimessi_guariti, :deceduti]), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "png")
	@time pplot.line_plot(select(df, [:totale_positivi, :totale_ospedalizzati, :dimessi_guariti, :deceduti]), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "pdf")
end

# TODO: to be tested
module test_ode
	include("ode.jl")
	include("params.jl")
	include("pplot.jl")

	df = model_params.get_data("data/italy/")
	u0,p,tspan = model_params.extract_params(df)

	prob = ode.get_ODE_problem(ode.SEIR, u0, tspan, p)
	@time sol = ode.get_solution(prob)

	pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "png")
	pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "pdf")
end

module test_abm
	using Agents
	include("params.jl")
	include("pplot.jl")
	include("graph.jl")

	df = model_params.get_data("data/italy/")
	abm_parameters = model_params.extract_params(df, 8, (50, 5000), 0.01)
	@show abm_parameters
	model = graph.init(; abm_parameters...)
	# @show model
	# ERROR: AssertionError: length(linewidths) == length(colors)
	@time pplot.video(model, graph.agent_step!, Agents.dummystep; title="graph_agent", path="img/video/")

	model = graph.init(; abm_parameters...)
	@time data = graph.collect(model, graph.agent_step!; n=100)
	@show data
	pplot.line_plot(data, df[1:101,:data], "img/abm/", "graph_agent", "png")
	pplot.line_plot(data, df[1:101,:data], "img/abm/", "graph_agent", "pdf")

	# @show 5 > rand(Uniform(4.8,5.3))
end

module test_controller
end