@time include("pplot.jl")
@time include("params.jl")

@time df = model_params.get_data()
@time p_gen = model_params.extract_params(df)
p = p_gen()

@time pplot.line_plot(df, "dpc-covid-19-italia")

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

# test sn_graph
include("social_network_graph.jl")
title = "social_network_graph_abm"
attractors = rand(3)
space_dimension = (200,200)
subspace_attr = (100,100)

# test behaviour to be similar to the ode one in a line graph
@time model = sn_graph.init(;
	N=100, space_dimension=space_dimension, attractors = attractors, 
	max_force = [1 + rand() for _ in 1:length(attractors)],
	attr_pos = [subspace_attr .* rand(2) for _ in 1:length(attractors)], 
	γ = p.γ, σ = p.σ, δ = p.δ₀, ω = p.ω, ϵ = p.ϵ, R₀ = p.R₀_n)
@time data = sn_graph.collect(model, sn_graph.agent_step!, sn_graph.model_step!)
@time pplot.line_plot(data, "prova")

# test the visual behaviour through a video
@time model = sn_graph.init(;
	N=100, space_dimension=space_dimension, attractors=attractors, 
	max_force = [1 + rand() for _ in 1:length(attractors)],
	attr_pos=[subspace_attr .* rand(2) for _ in 1:length(attractors)], 
	γ = p.γ, σ = p.σ, δ = p.δ₀, ω = p.ω, ϵ = p.ϵ, R₀ = p.R₀_n)
# FIXME: agents are not moving
@time pplot.record_video(model, sn_graph.agent_step!, sn_graph.model_step!; name="img/prova_", frames=1000)