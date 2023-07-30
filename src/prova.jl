using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using Zygote, Plots, OptimizationOptimJL

function seir!(du, u, p, t)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ, η, ξ = p
    μ = δ / 1111
    du[1] = μ * sum(u) - R₀ * γ * (1 - η) * S * I + ω * R - ξ * S - μ * S # dS
    du[2] = R₀ * γ * (1 - η) * S * I - σ * E - μ * E # dE
    du[3] = σ * E - γ * I - δ * I - μ * I # dI
    du[4] = (1 - δ) * γ * I - ω * R + ξ * S - μ * R # dR
    du[5] = δ * γ * I # dD
end

u0 = [0.9999, 0.0, 0.0001, 0.0, 0.0]
tspan = (0.0, 1200.0)
tsteps = 0.0:1.0:1200.0
p = [3.54, 1 / 14, 1 / 5, 1 / 240, 0.001, 0.0, 0.0]

prob = ODEProblem(seir!, u0, tspan, p)
sol_ = solve(prob, Tsit5())
plot(sol_)

# direct sensitivity analisys
prob = ODEForwardSensitivityProblem(seir!, u0, tspan, p)
sol = solve(prob, Tsit5())
x, dp = extract_local_sensitivities(sol)
dR₀ = dp[1]
dγ = dp[2]
dσ = dp[3]
dω = dp[4]
dδ = dp[5]
dη = dp[6]
dξ = dp[7]
size(dp)

function plot_sensitivity()
    plt = []
    push!(plt, plot(sol_, lw=2, title="Data", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dR₀', lw=2, title="Sensitivity to R₀", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dγ', lw=2, title="Sensitivity to γ", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dσ', lw=2, title="Sensitivity to σ", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dω', lw=2, title="Sensitivity to ω", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dδ', lw=2, title="Sensitivity to δ", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dη', lw=2, title="Sensitivity to η", titlefontsize=10, legend=false))
    push!(plt, plot(sol.t, dξ', lw=2, title="Sensitivity to ξ", titlefontsize=10, legend=false))
    plot(plt...)
end

plot_sensitivity()

# try to use optimization to find best parameters for η but failed
# stay with Ipopt, but wish to migrate to use NeuralODE as a controller
function loss(p)
    sol = solve(prob, Tsit5(), p=p, saveat=tsteps, verbose=false)
    loss = sum(abs2, sol[3, :] .- 0.1)
    return loss, sol
end

callback = function (p, l, pred)
    display(l)
    plt = plot(pred)
    display(plt)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p, lb=[3.3 - eps(), 1 / 14 - eps(), 1 / 5 - eps(), 1 / 240 - eps(), 0.001 - eps(), 0.0 - eps(), 0.0 - eps()], ub=[5.8 + eps(), 1 / 14 + eps(), 1 / 5 + eps(), 1 / 240 + eps(), 0.001 + eps(), 1.0, 1.0])

result = Optimization.solve(optprob, NelderMead(), callback=callback, maxiters=100)
result.u[6] = 0.7
remade = solve(remake(prob, p=result.u), Tsit5(), saveat=tsteps, verbose=false)
plot(remade)
