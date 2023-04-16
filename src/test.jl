module test_parameters
	include("params.jl")

	@time df = model_params.get_data()

	@time ode_parameters = model_params.extract_params(df)
	@time abm_parameters = model_params.extract_params(df, 8, (50,5000), 0.01)
end

module test_plot
	using DataFrames
	include("params.jl")
	include("pplot.jl")

	df = model_params.get_data()
	@time pplot.line_plot(select(df, [:variazione_totale_positivi]), 
		df[!, :data], "img/data_plot/", "rapporto-positivi-guariti", "png")
	@time pplot.line_plot(select(df, [:variazione_totale_positivi]), 
		df[!, :data], "img/data_plot/", "rapporto-positivi-guariti", "pdf")
	@time pplot.line_plot(select(df, [:nuovi_positivi, :dimessi_guariti, :deceduti]), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "png")
	@time pplot.line_plot(select(df, [:nuovi_positivi, :dimessi_guariti, :deceduti]), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "pdf")
end

module test_ode
	include("ode.jl")
	include("params.jl")
	include("pplot.jl")

	df = model_params.get_data()
	u0,p,tspan = model_params.extract_params(df)

	prob = ode.get_ODE_problem(ode.SEIR, u0, tspan, p)
	@time sol = ode.get_solution(prob)

	pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "png")
	pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "pdf")
end

module test_abm
	include("ode.jl")
	include("params.jl")
	include("pplot.jl")
	include("graph.jl")

	df = model_params.get_data()
	abm_parameters = model_params.extract_params(df, 8, (50, 5000), 0.01)
	model = graph.init(; abm_parameters...)
end

module test_controller
end