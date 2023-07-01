module ode
using OrdinaryDiffEq, DifferentialEquations, Random
using Plots, Distributions

rng = Xoshiro(1234)

function seir!(du, u, p, t)
    # https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown/ode_lockdown.md
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ = p
    dS = -R₀ * γ * S * I + ω * R
    dE = R₀ * γ * S * I - σ * E
    dI = σ * E - γ * I - δ * I
    dR = γ * I - ω * R
    dD = δ * I * γ
    du[1] = dS
    du[2] = dE
    du[3] = dI
    du[4] = dR
    du[5] = dD
end

function Fseir!(du, u, p, t)
    S, E, I, R, D = u
    R₀, γ, σ, ω, δ, Δ = p
    dS = -R₀ * γ * S * I + ω * R + Δ * S
    dE = R₀ * γ * S * I - σ * E + Δ * E
    dI = σ * E - γ * I - δ * I
    dR = γ * I - ω * R + Δ * R
    dD = δ * I * γ
    du[1] = dS
    du[2] = dE
    du[3] = dI
    du[4] = dR
    du[5] = dD
end

get_ode_problem(F, u, tspan, p) = ODEProblem(F, u, tspan, p)
get_ode_solution(prob; solver=Tsit5, saveat=1) = solve(prob, solver(), saveat=saveat)
get_ode_integrator(prob; solver=Tsit5) =
    OrdinaryDiffEq.init(prob, solver(); advance_to_stop=true)

# TODO: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269843#abstract0
# TODO: https://appliednetsci.springeropen.com/articles/10.1007/s41109-021-00378-3

function F!(du, u, p, t)
    S, E, I, R, D, H = u
    R₀, γ, σ, ω, δ, η, ξ, τ = p
    du[1] = (-R₀ * γ * (1 - η) * S * I) + (ω * R) - (S * ξ) # dS
    du[2] = (R₀ * γ * (1 - η) * S * I) - (σ * E) # dE
    du[3] = (σ * E) - (γ * I) - (δ * I) # dI
    du[4] = ((1 - δ) * γ * I - ω * R) + (S * ξ) # dR
    du[5] = (δ * I * γ) # dD
    du[6] = 0.0 # (H - (I * D)) * (1 - η) # dH
end

condition_vaccine(u, t, integrator) = rand(rng) < 1 / 365
condition_voc(u, t, integrator) = rand(rng) < 8 * 10e-4
condition_countermeasures(u, t, integrator) = integrator(t, Val{1})[3]

function affect_vaccine!(integrator)
    println("vaccine")
    integrator.p[7] = (1 - (1 / integrator.p[1])) / 0.83 * integrator.p[4]
end

function affect_voc!(integrator)
    println("voc")
    integrator.p[1] = rand(rng, Uniform(3.3, 5.7))
    integrator.p[2] = abs(rand(rng, Normal(integrator.p[2], integrator.p[2] / 5)))
    integrator.p[3] = abs(rand(rng, Normal(integrator.p[3], integrator.p[3])))
    integrator.p[4] = abs(rand(rng, Normal(integrator.p[4], integrator.p[4] / 10)))
    integrator.p[5] = abs(rand(rng, Normal(integrator.p[5], integrator.p[5] / 10)))
end

function affect_countermeasures!(integrator)
    # println("integrator: $(integrator.u[3]), $(integrator(integrator.u[3], Val{1}))")
end

vaccine_cb = ContinuousCallback(condition_vaccine, affect_vaccine!)
voc_cb = ContinuousCallback(condition_voc, affect_voc!)
countermeasures_cb = ContinuousCallback(condition_countermeasures, affect_countermeasures!)
cb = CallbackSet(vaccine_cb, voc_cb, countermeasures_cb)

pop = randexp(Xoshiro(1337), 20) * 1000
pop = map((x) -> round(Int, x), pop)
S = (sum(pop) - 1) / sum(pop)
E = 0.0
I = 1 / sum(pop)
R = 0.0
D = 0.0
H = 0.0
tspan = (1.0, 1200.0)
p = [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0, 0.0]
prob = ODEProblem(F!, [S, E, I, R, D, H], tspan, p, callback=cb)
sol = solve(prob, Tsit5(), saveat=1)
plot(sol, labels=["S" "E" "I" "R" "D" "Happiness"])

end
