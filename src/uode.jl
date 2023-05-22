module uode
# implement a simple ODE system and solver for seir model
# refers to. https://stackoverflow.com/questions/75902221/how-to-solve-the-error-undefvarerror-interpolatingadjoint-not-defined-using-d
using OrdinaryDiffEq

function seir!(du, u, p, t)
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

function get_ode_problem(F, u, tspan, p)
    return ODEProblem(F, u, tspan, p)
end

function get_ode_solution(prob)
    return solve(prob, Tsit5())
end
end
