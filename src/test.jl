module test_parameters
	using DataFrames, FileIO, JLD2
	include("params.jl")

	@time df = model_params.download_dataset("data/OWID/", "https://covid.ourworldindata.org/data/owid-covid-data.csv")
	@time data = model_params.dataset_from_location(df, "ITA")

	# LinearAlgebra.SingularException(3)
	# https://stackoverflow.com/questions/68967232/why-does-julia-fails-to-solve-linear-system-systematically
	@time sys, params = model_params.system_identification(select(data, Not([:date, :population, :reproduction_rate])))

	@time abm_parameters = model_params.get_abm_parameters(data, 20, 0.01)
	@time model_params.save_parameters(abm_parameters, "data/parameters/", "abm_parameters")
	load("data/parameters/abm_parameters.jld2")
end

module test_plot
	using DataFrames
	include("params.jl")
	include("pplot.jl")

	@time df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
	@time data = model_params.dataset_from_location(df, "ITA")

	@time pplot.line_plot(
		select(data, Not([:date, :population, :reproduction_rate])), 
		data[!, :date], "img/data_plot/", "explorative_plot", "pdf")
	@time pplot.line_plot(select(data, :reproduction_rate), data[!, :date], 
		"img/data_plot/", "reproduction_rate", "pdf")
end

module test_abm
	using Agents, DataFrames
	using Statistics: mean

	include("params.jl")
	include("pplot.jl")
	include("graph.jl")

	df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
	df = model_params.dataset_from_location(df, "ITA")
	abm_parameters = model_params.get_abm_parameters(df, 20, 0.01)

	@time model = graph.init(; abm_parameters...)
	@time data = graph.collect(model, graph.agent_step!, graph.model_step!; n=length(df[!,1])-1)
	pplot.line_plot(
		select(data, 
			Not([:happiness_happiness, :infected_detected, 
			:quarantined_detected, :recovered_detected])),
		df[!,:date], "img/abm/", "graph_agent", "pdf")
	pplot.line_plot(
		select(data, 
			[:infected_detected, :quarantined_detected]), 
			#:recovered_detected]),
		df[!,:date], "img/abm/", "graph_agent_countermeasures", "pdf")
	pplot.line_plot(
		select(data, [:happiness_happiness]),
		df[!,:date], "img/abm/", "graph_agent_happiness", "pdf")
	
	@time model = graph.init(; abm_parameters...)
	@time pplot.custom_video(model, graph.agent_step!, graph.model_step!; 
		title="graph_agent_custom", path="img/video/", 
		format=".mp4", frames=length(df[!,1])-1)
end

# TODO: to be implemented
module test_uode
	include("uode.jl")
	include("params.jl")
	include("pplot.jl")
end

module test_controller
	include("params.jl")
	include("pplot.jl")
	include("controller.jl")
	# il controllore dovrebbe prendere i dati dopo tot passi del collect e tirare le somme
	# successivamente modifica i parametri e fa riprendere il collect  e cosi via fino alla fine
end