@time include("pplot.jl")
@time include("params.jl")

@time include("ode.jl")
@time df = model_params.get_data()
@time p_gen = model_params.extract_params(df)

p = p_gen()
i = df[1,3]/p.N
r = df[1,1]/p.N
d = df[1,2]/p.N
s = (p.N-i-r-d)/p.N

a = [df[!,i] / p.N for i in 1:4]
@time pplot.line_plot(a, "dpc-covid-19-italia")

# test ODE model
u0 = [s, 0.0, i, r, d, p.R₀_n]
@time prob = ode.get_ODE_problem(ode.FODE, u0, (0, p.T), p)
@time sol = ode.get_solution(prob)
pplot.line_plot(sol, "SEIRD-dmodel")

# test SDE model
u0 = [s, 0.0, i, r, d, p.R₀_n, p.δ₀]
@time prob = ode.get_SDE_problem(ode.FSDE, ode.G, u0, (0, p.T), p)
@time sol = ode.get_solution(prob) #FIXME: Error: DimensionMismatch
pplot.line_plot(sol, "SIR-smodel")

include("social_network_graph.jl")
include("collect.jl")
# test sn_graph
@time model = sn_graph.init()
@time data = collect_data.collect(model, sn_graph.agent_step!, sn_graph.model_step!)
@time pplot.record_video(model,sn_graph.agent_step!, sn_graph.model_step!)
pplot.line_plot(data, "social_network_graph_abm")