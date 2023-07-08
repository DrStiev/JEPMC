using OrdinaryDiffEq, DifferentialEquations, Random
using Plots, Distributions

rng = Xoshiro(1234)

# TODO: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269843#abstract0
# TODO: https://appliednetsci.springeropen.com/articles/10.1007/s41109-021-00378-3

function F!(du, u, p, t)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ, η, ξ = p
    du[1] = (-R₀ * γ * (1 - η) * S * I) + (ω * R) - (S * ξ) # dS
    du[2] = (R₀ * γ * (1 - η) * S * I) - (σ * E) # dE
    du[3] = (σ * E) - (γ * I) - (δ * I) # dI
    du[4] = ((1 - δ) * γ * I - ω * R) + (S * ξ) # dR
    du[5] = (δ * I * γ) # dD
end

condition_vaccine(u, t, integrator) = rand(rng) < 1 / 365
condition_voc(u, t, integrator) = rand(rng) < 8e-3
condition_migrate(u, t, integrator) = true

function affect_vaccine!(integrator)
    println("vaccine")
    integrator.p[7] = (1 - (1 / integrator.p[1])) / 0.83 * integrator.p[4]
end

function affect_voc!(integrator)
    println("voc")
    integrator.p[1] = rand(rng, Uniform(3.3, 5.7))
    integrator.p[2] = abs(rand(rng, Normal(integrator.p[2], integrator.p[2] / 10)))
    integrator.p[3] = abs(rand(rng, Normal(integrator.p[3], integrator.p[3] / 10)))
    integrator.p[4] = abs(rand(rng, Normal(integrator.p[4], integrator.p[4] / 10)))
    integrator.p[5] = abs(rand(rng, Normal(integrator.p[5], integrator.p[5] / 10)))
end

function affect_migrate!(integrator)
    println("migrate")

end

vaccine_cb = ContinuousCallback(condition_vaccine, affect_vaccine!)
voc_cb = ContinuousCallback(condition_voc, affect_voc!)
migrate_cb = ContinuousCallback(condition_migrate, affect_migrate!)
cb = CallbackSet(vaccine_cb, voc_cb, migrate_cb)

pop = map((x) -> round(Int, x), randexp(Xoshiro(42), 9) * 10000)
param = [[(p - 1) / p, 0, 1 / p, 0, 0, 0, 0] for p in pop]
tspan = (1.0, 1200.0)
p = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0]
prob = [ODEProblem(F!, i, tspan, p, callback=cb) for i in param]; #, callback=cb)
sol = [solve(p, Tsit5()) for p in prob];
plt = []
length(sol)
for i in 1:length(sol)
    push!(plt, plot(sol[i].t ,sol[i][1:5, :]', title="Node with $(pop[i]) people", label=["S" "E" "I" "R" "D"], titlefontsize=6))
end
plot(plt...)

mean(sol)

for i in 1:length(sol)
    println(length(sol[i].t))
end

# function o(prob::ODEProblem)
#     sol = solve(prob, Tsit5())
#     return sol
# end

l = maximum(length, [sol[1].t, sol[2].t])

# https://github.com/epirecipes/sir-julia
# https://julialang.org/blog/2019/01/fluxdiffeq/
# https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DifferentialEquations, Lux, SciMLSensitivity, ComponentArrays, OptimizationOptimisers

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length=datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat=tsteps))

dudt2 = Lux.Chain(x -> x .^ 3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot=true)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, ode_data[1, :], label="data")
        scatter!(plt, tsteps, pred[1, :], label="prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob,
    ADAMW(),
    callback=callback,
    maxiters=300
)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false
)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)

using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq,
    Plots, CUDA, SciMLSensitivity, Random, ComponentArrays
import DiffEqFlux: NeuralODE
CUDA.allowscalar(false) # Makes sure no slow operations are occuring

#rng for Lux.setup
rng = Random.default_rng()
# Generate Data
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length=datasize)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Make the data into a GPU-based array if the user has a GPU
ode_data = Lux.gpu(solve(prob_trueode, Tsit5(), saveat=tsteps))


dudt2 = Lux.Chain(x -> x .^ 3, Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
u0 = Float32[2.0; 0.0] |> Lux.gpu
p, st = Lux.setup(rng, dudt2)
p = p |> ComponentArray |> Lux.gpu
st = st |> Lux.gpu

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)

function predict_neuralode(p)
    Lux.gpu(first(prob_neuralode(u0, p, st)))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
# Callback function to observe training
list_plots = []
iter = 0
callback = function (p, l, pred; doplot=true)
    global list_plots, iter
    if iter == 0
        list_plots = []
    end
    iter += 1
    display(l)
    # plot current prediction against data
    plt = scatter(tsteps, Array(ode_data[1, :]), label="data")
    scatter!(plt, tsteps, Array(pred[1, :]), label="prediction")
    push!(list_plots, plt)
    if doplot
        display(plot(plt))
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
result_neuralode = Optimization.solve(optprob, ADAMW(0.05), callback=callback, maxiters=300)
