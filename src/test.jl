module test_parameters
using DataFrames
include("params.jl")

model_params.download_dataset(
    "data/OWID/",
    "https://covid.ourworldindata.org/data/owid-covid-data.csv",
)
df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")

# LinearAlgebra.SingularException(3)
@time sys, params = model_params.system_identification(
    float.(Array(day_info))',
    float.((1:length(day_info[!, 1]))),
)

abm_parameters = model_params.get_abm_parameters(20, 0.01)
model_params.save_parameters(abm_parameters, "data/parameters/", "abm_parameters")
params = model_params.load_parameters("data/parameters/abm_parameters.jld2")
end

module test_plot
using DataFrames, Plots
include("params.jl")
include("pplot.jl")

df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")

p = plot(
    plot(
        Array(day_info),
        labels=["Infected" "Tests" "Vaccinations" "Deaths"],
        title="Detected Dynamics",
    ),
    plot(
        Array(total_count),
        labels=["Susceptible" "Infected" "Deaths" "Tests"],
        title="Overall Dynamics",
    ),
    plot(Array(R₀), labels="R₀", title="Reproduction Rate"),
)

pplot.save_plot(p, "img/data_plot/", "cumulative_plot", "pdf")
end

module test_abm
using Agents, DataFrames, Plots, Distributions, Random
using Statistics: mean

include("params.jl")
include("pplot.jl")
include("graph.jl")

df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")

abm_parameters = model_params.get_abm_parameters(20, 0.01, 3300)
model = graph.init(; abm_parameters...)

data = graph.collect(model; n=length(date[!, 1]) - 1, controller_step=7)
graph.save_dataframe(data, "data/abm/", "ABM SEIR")
df = model_params.read_local_dataset("data/abm/ABM SEIR_2023-05-28.csv")

p1 = select(
    data,
    [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
)
p2 = select(data, [:active_countermeasures])
p3 = select(data, [:happiness_happiness])
p4 = select(data, [:R₀])

p = plot(
    plot(
        Array(p1),
        labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
        title="ABM Full Dynamics",
    ),
    plot(Array(p2), labels="η", title="Countermeasures strickness"),
    plot(Array(p3), labels="Happiness", title="Cumulative Happiness"),
    plot(Array(p4), labels="R₀", title="Reproduction number"),
)
pplot.save_plot(p, "img/abm/", "cumulative_plot", "pdf")

# sys, params = model_params.system_identification(Array(p1)', (1:length(p1[!,1])))

# TODO: SISTEMAMI
# model = graph.init(; abm_parameters...)
# pplot.custom_video(
#     model,
#     graph.agent_step!,
#     graph.model_step!;
#     title="graph_abm",
#     path="img/video/",
#     format=".mp4",
#     frames=length(date[!, 1]) - 1
# )
end

module test_uode
using Plots, DataFrames, Random, Distributions
include("uode.jl")
include("params.jl")
include("pplot.jl")

# must be between [0-1] otherwise strange behaviour
u, p, t = model_params.get_ode_parameters()
prob = uode.get_ode_problem(uode.seir!, u, t, p)
sol = uode.get_ode_solution(prob)

p = plot(
    sol,
    labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
    title="SEIR Dynamics",
)
pplot.save_plot(p, "img/ode/", "cumulative_plot", "pdf")

# LinearAlgebra.SingularException(3)
# sys, params = model_params.system_identification(Array(sol), sol.t)
end

module test_controller
# https://link.springer.com/article/10.1007/s40313-023-00993-8
using Agents, DataFrames, Plots, Distributions, Random
using Statistics: mean

include("params.jl")
include("pplot.jl")
include("graph.jl")
include("controller.jl")

df = model_params.read_local_dataset("data/OWID/owid-covid-data.csv")
date, day_info, total_count, R₀ = model_params.dataset_from_location(df, "ITA")

abm_parameters = model_params.get_abm_parameters(20, 0.001, 3300)
@time model = graph.init(; abm_parameters...)

data = graph.collect(model, graph.agent_step!, graph.model_step!; n=30)
select(data, [:infected_detected, :controls])
model.step_count
model.properties

end

module to_be_implemented
# creo una matrice di spostamento tra i vari nodi
# creo un grafo come Graph(M+M')
using Graphs, GraphPlot, Random, LinearAlgebra
using Distributed
include("uode.jl")
include("params.jl")

# cero una matrice di migrazione e un insieme
# di valori da associare ai nodi di un grafo
function get_migration_matrix(travel_rate, C, avg)
    rng = Xoshiro(1337)
    pop = randexp(rng, C) * avg
    number_point_of_interest = map((x) -> round(Int, x), pop)
    migration_rate = zeros(C, C)
    for c = 1:C
        for c2 = 1:C
            migration_rate[c, c2] =
                (number_point_of_interest[c] + number_point_of_interest[c2]) /
                number_point_of_interest[c]
        end
    end
    maxM = maximum(migration_rate)
    migration_rate = (migration_rate .* travel_rate) ./ maxM
    migration_rate[diagind(migration_rate)] .= 1.0
    migration_rate_sum = sum(migration_rate, dims=2)
    for c = 1:C
        migration_rate[c, :] ./= migration_rate_sum[c]
    end
    return migration_rate, number_point_of_interest
end

C = 20
migration_matrix, nodes_population = get_migration_matrix(0.01, C, 3300)
1 - (migration_matrix[1, 1] + sum(migration_matrix[1, 2:end]))
out = migration_matrix[1, 2:end] .* nodes_population[1]
i = migration_matrix[2:end, 1] .* nodes_population[2:end]
sum(i) - sum(out)
nodes_population
pop = sum(nodes_population)
# parametri ode relativi all'epidemia
p = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007]

# creo matrice relativa al grafo 
posPatientZero = rand(1:C)
# N, S, E, I, R, D
seirMatrix = zeros(Float64, C, 6)
for c in 1:C
    seirMatrix[c, :] = [nodes_population[c], 1.0, 0.0, 0.0, 0.0, 0.0]
    if c == posPatientZero
        seirMatrix[c, :] = [nodes_population[c], (nodes_population[c] - 1) / nodes_population[c], 0.0, 1 / nodes_population[c], 0.0, 0.0]
    end
end
seirMatrix

# differenza tra individui che entrano nel nodo i e individui che escono dal nodo i 
# calcolo semplicistico ← Δ
Δ = sum(migration_matrix[:, 1]) - sum(migration_matrix[1, :])
t = (0.0, 1.0)
push!(p, Δ)
prob = uode.get_ode_problem(uode.Fseir!, seirMatrix[1, 2:end], t, p)
sol = uode.get_ode_solution(prob)
prob = uode.get_ode_problem(uode.Fseir!, sol.u[end], t, p)
sol = uode.get_ode_solution(prob)

# distributed computation to calculate seir
addprocs(nprocs())
for i in 0.0:1.0:10.0 # step to compute ode
    t = (i, i + 1.0)
    @distributed for N in 1:C
        println("The N of this iteration in $N calculate node $(nodes_population[N])")
        prob = uode.get_ode_problem(uode.Fseir!, seirMatrix[N, :], t, p)
        sol = uode.get_ode_solution(prob)
        seirMatrix[N, 2:end] = sol.u[end]
    end
end

p = plot(
    sol,
    labels=["Susceptible" "Exposed" "Infected" "Recovered" "Dead"],
    title="SEIR Dynamics",
)
end
