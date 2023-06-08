# SciML Tools

# downgrade OrdinaryDiffEq and Optimization to solve include error but
# still new error arise
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, DataFrames

gr()

include("utils.jl")
include("uode.jl")
include("graph.jl")

# Define the experimental parameter
param = parameters.get_abm_parameters(20, 0.01, 3300)
model = graph.init(; param...)
data = graph.collect(model; n = 30, showprogress = true)
ddata = select(
    data,
    [:susceptible_status, :exposed_status, :infected_status, :recovered_status, :dead],
)

tspan = float.([i for i = 1:size(Array(ddata), 1)])
u, p_, _ = parameters.get_ode_parameters(20, 3300)

X =
    Array(
        select(
            data,
            [
                :susceptible_status,
                :exposed_status,
                :infected_status,
                :recovered_status,
                :dead,
            ],
        ),
    )' ./ sum(data[1, :])

plot(X', xlabel = "time", ylabel = "individuals", labels = ["S" "E" "I" "R" "D"])

rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
U = Lux.Chain(
    Lux.Dense(size(X, 1), 8, rbf),
    Lux.Dense(8, 8, rbf),
    Lux.Dense(8, 8, rbf),
    Lux.Dense(8, size(X, 1)),
)
# Get the initial parameters and state variables of the model
p, st = Lux.setup(StableRNG(1234), U)

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p, st)[1] # network prediction
    du[1] = û[1]
    du[2] = û[2]
    du[3] = û[3]
    du[4] = û[4]
    du[5] = û[5]
end

# function ude_dynamics!(du, u, p, t, p_true)
#     û = U(u, p, st)[1] # Network prediction
#     du[1] = -p_true[1] * u[1] * p_true[2] * u[3] + û[4]
#     du[2] = p_true[1] * u[1] * p_true[2] * u[3] - û[2]
#     du[3] = û[2] - p_true[2] * u[3] + û[3]
#     du[4] = p_true[2] * u[3] - û[4]
#     du[5] = p_true[2] * u[3] * p_true[5]
# end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t)#, p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!, X[:, 1], (tspan[1], tspan[end]), p)

function predict(θ, X = X[:, 1], T = tspan)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T, abstol = 1e-6, reltol = 1e-6))
end

function loss(θ)
    X̂ = predict(θ)
    mean(abs2, X .- X̂)
end

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# seems work but need test (heavy memory consuming need kos)
res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 2000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 400)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u

ts = first(tspan):(mean(diff(tspan))/2):last(tspan)
X̂ = predict(p_trained, X[:, 1], ts)
# Neural network guess
Ŷ = U(X̂, p_trained, st)[1]

## Analysis of the trained network
# Plot the data and the approximation
# Trained on noisy data vs real solution
pl_trajectory = plot(
    ts,
    transpose(X̂),
    xlabel = "t",
    ylabel = "x(t), y(t)",
    color = :red,
    label = ["UDE Approximation" nothing],
)
scatter!(tspan, transpose(X), color = :black, label = ["Measurements" nothing])

# Symbolic regression via sparse regression (SINDy based)
nn_problem = DirectDataDrivenProblem(X̂, Ŷ)
λ = exp10.(-3:0.01:3)
opt = ADMM(λ)
@variables u[1:size(X, 1)]
b = polynomial_basis(u, size(X, 1))
basis = Basis(b, u);

options = DataDrivenCommonOptions(
    maxiters = 10_000,
    normalize = DataNormalization(ZScoreTransform),
    selector = bic,
    digits = 1,
    data_processing = DataProcessing(
        split = 0.9,
        batchsize = 30,
        shuffle = true,
        rng = StableRNG(1234),
    ),
)

nn_res = solve(nn_problem, basis, opt, options = options)
nn_eqs = get_basis(nn_res)
println(nn_res)

# Define the recovered, hybrid model
function recovered_dynamics!(du, u, p, t)
    û = eqs(u, p) # Recovered equations
    du[1] = -p_[1] * u[1] * p_[2] * u[3] + û[1]
    du[2] = p_[1] * u[1] * p_[2] * u[3] - û[2]
    du[3] = û[2] - p_[2] * u[3] + û[3]
    du[4] = p_[2] * u[3] - û[1]
    du[5] = p_[2] * u[3] * p_[5]
end

estimation_prob = ODEProblem(recovered_dynamics!, u, tspan, get_parameter_values(nn_eqs))
estimate = solve(estimation_prob, Tsit5())

# Plot
plot(Array(actual_data))
plot!(estimate)

function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
    sum(abs2, Ŷ .- Y)
end

optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters = 1000)

# Look at long term prediction
t_long = (0.0, tspan[end] * 2)
estimation_prob = ODEProblem(recovered_dynamics!, u, t_long, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(estimate_long)
