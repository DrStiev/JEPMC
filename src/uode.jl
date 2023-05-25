module uode
using OrdinaryDiffEq

# TODO: secondo me si può realizzare l'idea di usare
# le ode per la descrizione dei passi del modello
# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/

# TODO: implementami
# ho un modello semplice che cerco di perfezionare
# utilizzando https://docs.sciml.ai/Overview/stable/showcase/missing_physics/
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

# TODO: https://docs.sciml.ai/Overview/stable/showcase/symbolic_analysis/

# TODO: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269843#abstract0
# TODO: https://appliednetsci.springeropen.com/articles/10.1007/s41109-021-00378-3
end
