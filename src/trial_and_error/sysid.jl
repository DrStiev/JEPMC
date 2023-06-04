using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra
using DataFrames

include("utils.jl")
include("uode.jl")
include("graph.jl")

u, p, t = parameters.get_ode_parameters(20, 3300)
prob = uode.get_ode_problem(uode.seir!, u, t, p)
sol = uode.get_ode_solution(prob)
sol.u
sol.t
df = DataFrame(sol.u, :auto)

plot(
    sol,
    xlabel = "time",
    ylabel = "percentage of individuals",
    labels = ["S" "E" "I" "R" "D"],
)

param = parameters.get_abm_parameters(20, 0.01, 3300)
model = graph.init(; param...)
data = graph.collect(model; n = 100, showprogress = true)
ddata = select(
    data,
    [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
)
X = DataFrame(float.(Array(ddata)'), :auto) ./ 58558.0 # normalization and numerical stability
t = float.([i for i = 1:size(X, 2)])

plot(
    Array(X)',
    xlabel = "time",
    ylabel = "percentage of individuals",
    labels = ["S" "E" "I" "R" "D"],
)

# use of the direct result of an ODEProblem
ddprob = ContinuousDataDrivenProblem(sol, sol.t)
@variables t s(t) e(t) i(t) r(t) d(t)
u = [s; e; i; r; d]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))

# plot the result
plot(plot(ddprob), plot(ddsol))

# obtain the results
sys = get_basis(ddsol)
params = get_parameter_map(sys)
println(sys)
println(params)

# use of the result of an ODEProblem after transform it to a DataFrame
ddprob = ContinuousDataDrivenProblem(Array(df), sol.t) # define the problem
@variables t s(t) e(t) i(t) r(t) d(t) # define the variables involved
u = [s; e; i; r; d]
basis = Basis(polynomial_basis(u, 5), u, iv = t) # construct a Basis
opt = STLSQ(exp10.(-5:0.1:-1)) # define the optimization algorithm
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))

# plot the result
plot(plot(ddprob), plot(ddsol))

# obtain the results
sys = get_basis(ddsol)
params = get_parameter_map(sys)
println(sys)
println(params)

# use of the result of an Agent, so it's a DataFrame
# hardcoding the number of variables
ddprob = ContinuousDataDrivenProblem(Array(X), t) # define the problem
@variables t s(t) e(t) i(t) r(t) d(t) # define the variables involved
u = [s; e; i; r; d]
basis = Basis(polynomial_basis(u, 5), u, iv = t) # construct a Basis
opt = STLSQ(exp10.(-5:0.1:-1)) # define the optimization algorithm
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))

# plot the result
plot(plot(ddprob), plot(ddsol))

# obtain the results
sys = get_basis(ddsol)
params = get_parameter_map(sys)
println(sys)
println(params)

# use of the result of Agents.jl, so it's a DataFrame 
# not hardcoding the number of variables
ddprob = ContinuousDataDrivenProblem(Array(X), t) # define the problem
@variables t (u(t))[1:(size(X))[1]]
b = []
for i = 1:size(X)[1]
    push!(b, u[i])
end
basis = Basis(polynomial_basis(b, 5), u, iv = t) # construct a Basis
opt = STLSQ(exp10.(-5:0.1:-1)) # define the optimization algorithm
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))

# plot the result
plot(plot(ddprob), plot(ddsol))

# obtain the results
sys = get_basis(ddsol)
params = get_parameter_map(sys)
println(sys)
println(params)
