using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra
using DataFrames

include("utils.jl")
include("uode.jl")
# include("graph.jl")

u, p, t = parameters.get_ode_parameters(20, 3300)
prob = uode.get_ode_problem(uode.seir!, u, t, p)
sol = uode.get_ode_solution(prob)
sol.u
sol.t
df = DataFrame(sol)

# param = parameters.get_abm_parameters(20, 0.01, 3300)
# model = graph.init(; param...)
# data = graph.collect(model; n=100, showprogress=true)
# ddata = float.(Array(select(data, [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead]))')
# t = (1.0:length(data[!, 1]))

# funziona sse gli si passa il risultato di una ODE altrimenti no
# non credo questo possa essere utilizzato effettivamente
ddprob = ContinuousDataDrivenProblem(sol, sol.t)
@variables t s(t) e(t) i(t) r(t) d(t)
u = [s; e; i; r; d]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))

# BoundsError: attempt to access 6-element Vector{Float64} at index [7]
ddprob = ContinuousDataDrivenProblem(Array(df), sol.t)
@variables t s(t) e(t) i(t) r(t) d(t)
u = [s; e; i; r; d]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))

ddprob = ContinuousDataDrivenProblem(Array(df)', sol.t)
@variables t s(t) e(t) i(t) r(t) d(t)
u = [s; e; i; r; d]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
# ERROR: BoundsError: attempt to access 5-element Vector{SymbolicUtils.BasicSymbolic{Real}} at index [6]
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
