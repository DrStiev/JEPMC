using DataFrames, Agents

@time include("pplot.jl")
@time include("params.jl")

@time df = model_params.get_data("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
@time p_gen = model_params.extract_params(df)
p = p_gen()

# TODO: make a decent plot
# @time pplot.line_plot(df, "dpc-covid-19-italia")

# test ODE model
@time include("ode.jl")

e = 0.0
i = df[1,3]
r = df[1,1]
d = df[1,2]
s = (1.0-e-i-r-d)

u0 = [s, e, i, r, d, p.R₀_n, p.δ₀]
@time prob = ode.get_ODE_problem(ode.F, u0, (0, p.T), p)
@time sol = ode.get_solution(prob)
pplot.line_plot(sol[!, 2:6], "SEIRD-model")

# test su graph agent
@time include("graph.jl")
title = "graph_space_abm"
@time model = graph.init(;
	number_point_of_interest=p.number_point_of_interest,
	migration_rate=p.migration_rate,
	R₀=p.R₀_n, R̅₀=p.R̅₀, ψ=p.ψ, η=p.η, ξ=p.ξ, θ=p.θ, γ=p.γ, σ=p.σ, ω=p.ω, δ=p.δ₀, ϵ=p.ϵ)
# FIXME: plot is nonsense
@time data = graph.collect(model, graph.agent_step!, Agents.dummystep; n=p.T)
@time pplot.line_plot(data, title)

# TODO: add video rec

# test sn_graph
include("social_network_graph.jl")
title = "social_network_graph_abm"
attractors = rand(1)
space_dimension = (100,100)
max_force = [1 + rand() for _ in 1:length(attractors)]
attr_pos = [space_dimension .* rand(2) for _ in 1:length(attractors)]

# test behaviour to be similar to the ode one in a line graph
@time model = sn_graph.init(;
	N=100, space_dimension=space_dimension, attractors = attractors, 
	max_force = max_force, attr_pos = attr_pos,
	γ = p.γ, σ = p.σ, δ = p.δ₀, ω = p.ω, ϵ = p.ϵ, R₀ = p.R₀_n)
@time data = sn_graph.collect(model, sn_graph.agent_step!, sn_graph.model_step!; n=p.T)
@time pplot.line_plot(data, title)

# test the visual behaviour through a video
@time model = sn_graph.init(;
	N=100, space_dimension=space_dimension, attractors=attractors, 
	max_force = max_force, attr_pos = attr_pos,
	γ = p.γ, σ = p.σ, δ = p.δ₀, ω = p.ω, ϵ = p.ϵ, R₀ = p.R₀_n)
@time pplot.record_video(model, sn_graph.agent_step!, sn_graph.model_step!; name="img/"*title*"_", frames=p.T)

# test if the parameters of the model are saved properly
pplot.save_parameters(model, title)