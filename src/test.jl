module test_parameters
	using DataFrames
	include("params.jl")

	@time df = model_params.get_data("data/italy/")
	# need to start using owid data
	@time df_alt = model_params.get_data("data/OWID/", "https://covid.ourworldindata.org/data/owid-covid-data.csv")

	data = select(df, [:nuovi_positivi, :isolamento_domiciliare, :dimessi_guariti, :deceduti])
	# InexactError: Int64(750.4522263578023)
	@time sys, params = model_params.system_identification(data)

	@time abm_parameters = model_params.extract_params(df, 20, 0.01)
	@time ode_parameters = model_params.extract_params(df)
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
end

module test_abm
	using Agents, DataFrames, Random, Plots
	using Statistics: mean
	using FileIO, JLD2

	include("params.jl")
	include("pplot.jl")
	include("graph.jl")

	df = model_params.read_data()
	abm_parameters = model_params.extract_params(df, 20, 0.01, 2000)

	@time model = graph.init(; abm_parameters...)
	@time data = graph.collect(model, graph.agent_step!, graph.model_step!; n=length(df[!,1])-1)
	pplot.line_plot(
		select(data, Not([:happiness_happiness, :infected_detected, :quarantined_detected, :recovered_detected])),
		df[1:length(data[!,1]),:data], "img/abm/", "graph_agent", "pdf")
	pplot.line_plot(
		select(data, [:infected_detected, :quarantined_detected, :recovered_detected]),
		df[1:length(data[!,1]),:data], "img/abm/", "graph_agent_countermeasures", "pdf")
	pplot.line_plot(
		select(data, [:happiness_happiness]),
		df[1:length(data[!,1]),:data], "img/abm/", "graph_agent_happiness", "pdf")
	@time model = graph.init(; abm_parameters...)
	@time pplot.custom_video(model, graph.agent_step!, graph.model_step!; title="graph_agent_custom", path="img/video/", format=".mp4", frames=length(df[!,1])-1)

	model_params.save_parameters(model.properties, "data/parameters/", "abm_parameters")
	load("data/parameters/abm_parameters.jld2")
end

module test_controller
	include("params.jl")
	include("pplot.jl")
	include("controller.jl")
	# il controllore dovrebbe prendere i dati dopo tot passi del collect e tirare le somme
	# successivamente modifica i parametri e fa riprendere il collect  e cosi via fino alla fine
end