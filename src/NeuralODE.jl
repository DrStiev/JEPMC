# https://github.com/epirecipes/sir-julia
# https://julialang.org/blog/2019/01/fluxdiffeq/
# https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
# https://docs.sciml.ai/DiffEqFlux/stable/examples/GPUs/
# https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/
# https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/
# https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DifferentialEquations, Lux, SciMLSensitivity, ComponentArrays, OptimizationOptimisers

rng = Xoshiro(1234)
pop = 1e6
u = [(pop - 1) / pop, 0, 1 / pop, 0, 0]
datasize = 171 # ≈ 1 entrata a settimana
tspan = (0.0f0, 1200.0f0)
param = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0]
tsteps = range(tspan[1], tspan[2], length=datasize)

prob_trueode = ODEProblem(F!, u, tspan, param, callback=cb)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat=tsteps))

dudt2 = Lux.Chain(
    Lux.Dense(size(ode_data, 1), 50, tanh),
    Lux.Dense(50, size(ode_data, 1)))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u, p, st)[1])
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
        plt = scatter(ode_data', label=["data S" "data E" "data I" "data R" "data D"])
        plot!(plt, pred', label=["prediction S" "prediction E" "prediction I" "prediction R" "prediction D"])
        # scatter!(plt, pred', label="prediction")
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
    ADAM(0.05),
    callback=callback,
    maxiters=1_000
)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false
)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)
