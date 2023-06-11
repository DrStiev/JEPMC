module uode
using OrdinaryDiffEq

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
get_ode_solution(prob) = solve(prob, Tsit5())
get_ode_integrator(prob) = OrdinaryDiffEq.init(prob, Tsit5(); advance_to_stop = true)

# TODO: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269843#abstract0
# TODO: https://appliednetsci.springeropen.com/articles/10.1007/s41109-021-00378-3
end
