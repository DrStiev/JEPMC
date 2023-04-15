using DataFrames

@time include("pplot.jl")
@time include("params.jl")

# test parameters creation
@time df = model_params.get_data("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
@time p_gen = model_params.extract_params(df)
p = p_gen()

# test plot function
@time pplot.line_plot(select(df, [:variazione_totale_positivi]), "rapporto-positivi-guariti")
@time pplot.line_plot(select(df, [:totale_positivi, :dimessi_guariti, :deceduti]), "dpc-covid19-ita-andamento-nazionale")

# test ODE model
@time include("ode.jl")
@time p_gen = model_params.extract_params(df)

# test R₀ fisso
p = p_gen()

e = 0.0/p.N
i = df[1,:totale_positivi]/p.N
r = df[1,:dimessi_guariti]/p.N
d = df[1,:deceduti]/p.N
s = (1.0-e-i-r-d)

u0 = [s, e, i, r, d, p.R₀_n, p.δ₀]

@time prob = ode.get_ODE_problem(ode.F, u0, (0, p.T), p)
@time sol = ode.get_solution(prob)
pplot.line_plot(sol[!, 2:6], "SEIRD-model_with_fixed_R₀_at_1.6")

# test R₀ variabile
# poco infettivo molto mortale -> medio - medio -> tanto - poco ->
R̅₀(t,p) = t < 200 ? 1.6 : t < 600 ? 2.0 : 3.0
p = p_gen(R̅₀=R̅₀, R₀_n=1.0)
u0 = [s, e, i, r, d, p.R₀_n, p.δ₀]

@time prob = ode.get_ODE_problem(ode.F, u0, (0, p.T), p)
@time sol = ode.get_solution(prob)
pplot.line_plot(sol[!, 2:6], "SEIRD-model_with_variable_R₀")

# add randomness
R̅₀(t,p) = t < 200 ? 1.6 : t < 600 ? 2.0 : 3.0
p = p_gen(R̅₀=R̅₀, R₀_n=1.0, η=1.0/20)

@time sprob = ode.get_SDE_problem(ode.F, ode.G, u0, (0, p.T), p)
@time ssol = ode.get_solution(sprob, 100)
pplot.line_plot(ssol, "SEIRD-model_with_randomness_and_variable_R₀")

# test su graph agent
@time include("graph.jl")
title = "graph_space_abm"
@time p_gen = model_params.extract_params(df; C=8, min_max_population=(50,5000), max_travel_rate=0.01)

# test model with ode suppport
p = p_gen()
@time model = graph.init(;
	number_point_of_interest=p.number_point_of_interest,
	migration_rate=p.migration_rate, params=p)
@time data = graph.collect(model, graph.migrate!, graph.model_step!, p.T)
@time pplot.line_plot(data, title*"_with_ode")

# test model with ode support and variable R₀
R̅₀(t,p) = t < 200 ? 1.6 : t < 600 ? 2.0 : 3.0
p = p_gen(R̅₀=R̅₀, R₀_n=1.0, η=1.0)
@time model = graph.init(;
	number_point_of_interest=p.number_point_of_interest,
	migration_rate=p.migration_rate, params=p)
@time data = graph.collect(model, graph.migrate!, graph.model_step!, p.T)
@time pplot.line_plot(data, title*"_with_ode_and_variable_R₀")

# test model and record short video
p = p_gen()
@time model = graph.init(;
	number_point_of_interest=p.number_point_of_interest,
	migration_rate=p.migration_rate, params=p)
@time abmobs = graph.get_observable(model)
@time pplot.record_video(abmobs, model, title, 100)

# test model and plot result
p = p_gen()
@time model = graph.init(;
	number_point_of_interest=p.number_point_of_interest,
	migration_rate=p.migration_rate, params=p)
@time data = graph.collect(model, graph.agent_step!; n=p.T)
@time pplot.line_plot(data, model, title)