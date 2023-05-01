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

	population = 58_850_717
	df = model_params.read_data()
	d = [population - sum([df[i,:totale_positivi], df[i,:dimessi_guariti], df[i,:deceduti]]) for i in 1:length(df[!,:dimessi_guariti])]
	d = DataFrame([d], [:suscettibili])
	@time pplot.line_plot(select(df, [:nuovi_positivi, :isolamento_domiciliare]), 
		df[!, :data], "img/data_plot/", "rapporto_nuovi_positivi_quarantena", "pdf")	
	@time pplot.line_plot(hcat(d, select(df, [:totale_positivi, :dimessi_guariti, :deceduti])), 
		df[!, :data], "img/data_plot/", "dpc-covid19-ita-andamento-nazionale", "pdf")
	dtct = DataFrame([df[!, :totale_casi]./df[!, :tamponi]], [:rapporto_positivi_tamponi])
	@time pplot.line_plot(select(df, [:totale_casi, :tamponi]), df[!, :data], "img/data_plot/", "rapporto_positivi_tamponi", "pdf")
end

# TODO: to be implemented
module test_uode
	include("uode.jl")
	include("params.jl")
	include("pplot.jl")

	# df = model_params.get_data("data/italy/")
	df = model_params.read_data()

	S0 = 14e6
	u0 = [0.9*S0, 0.0, 0.0, 0.0, S0, 0.0, 0.0]
	# 		F,    β0,      α,     κ,      μ,   σ,   γ,   d,    λ
	p_ = [10.0, 0.5944, 0.4239, 1117.3, 0.02, 1/3, 1/5, 0.2, 1/11.2]
	tspan = (0.0, 21.0)
	tspan2 = (0.0,60.0)
	# TODO: test!
	pred = uode.get_prediction(u0, p_, tspan, tspan2)
	# u0, p, tspan = model_params.extract_params(df)
	# prob = ode.get_ODE_problem(ode.SEIR, u0, tspan, p)
	# @time sol = ode.get_solution(prob)
	
	# pplot.line_plot(sol, df[!,:data], "img/ode/", "seir_model", "pdf")
end

module test_abm
	using Agents, DataFrames, Random, Plots
	using Statistics: mean
	include("params.jl")
	include("pplot.jl")
	include("graph.jl")

	population = 2500.0 
	df = model_params.read_data()
	abm_parameters = model_params.extract_params(df, 20, population, 0.01)
	
	@time model = graph.init(; abm_parameters...)
	@time pplot.custom_video(model, graph.agent_step!, graph.model_step!; title="graph_agent_custom", path="img/video/", format=".mp4", frames=abm_parameters[:T])
	@time model = graph.init(; abm_parameters...)
	@time data = graph.collect(model, graph.agent_step!, graph.model_step!; n=abm_parameters[:T]-1)
	pplot.line_plot(select(data, Not([:happiness_happiness, :infected_detected, :quarantined_detected, :recovered_detected])), 
		df[1:length(data[!,1]),:data], "img/abm/", "graph_agent", "pdf")
	pplot.line_plot(select(data, [:infected_detected, :quarantined_detected, :recovered_detected]), 
		df[1:length(data[!,1]),:data], "img/abm/", "graph_agent_countermeasures", "pdf")
	pplot.line_plot(select(data, [:happiness_happiness]), 
		df[1:length(data[!,1]),:data], "img/abm/", "graph_agent_happiness", "pdf")
end

module test_controller
	include("params.jl")
	include("pplot.jl")
	include("controller.jl")
	# il controllore dovrebbe prendere i dati dopo tot passi del collect e tirare le somme
	# successivamente modifica i parametri e fa riprendere il collect  e cosi via fino alla fine
end