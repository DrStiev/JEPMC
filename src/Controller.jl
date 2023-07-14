using JuMP, Ipopt, Plots, InfiniteOpt
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_ftc_jump/function_map_ftc_jump.md
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_vaccine_jump/function_map_vaccine_jump.md
# questo esempio potrebbe essere buono per la NeuralODE
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_optimization/ode_lockdown_optimization.md

# PROVA QUESTO
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_infiniteopt/ode_lockdown_infiniteopt.md
# https://github.com/epirecipes/sir-julia
function controller!(
    initial_condition::Vector{Float64},
    parameters::Vector{Float64},
    η_max::Float64=0.5,
    I_max::Float64=0.1, # piu' basso e' I_max piu' alto dovra' essere η_max
    D_max::Float64=0.001;
    silent::Bool=true,
    timeframe::Tuple{Float64,Float64}=(0.0, 100.0), # anche il tempo potrebbe essere importante
    δt::Float64=0.1,
    showplot::Bool=false
)

    T = Int(timeframe[2] / δt)
    # specify a model using JuMP.Model passing an optimizer
    model = Model(Ipopt.Optimizer)

    # we declare the number of timesteps and vectors of our model variables
    # including the intervention level each timesteps+1 long
    # we also define the total cost of the intervention
    @variable(model, S[1:(T+1)])
    @variable(model, E[1:(T+1)])
    @variable(model, I[1:(T+1)])
    @variable(model, R[1:(T+1)])
    @variable(model, D[1:(T+1)])
    @variable(model, η[1:(T+1)])
    @variable(model, η_total)

    # we constrain our variables to be at their initial conditions
    # for the first element of the array and between 0 and 1 for the others
    # with the exception for the proportion of infected individuals,
    # which is constrained to be less than I_max
    @constraint(model, S[1] == initial_condition[1])
    @constraint(model, E[1] == initial_condition[2])
    @constraint(model, I[1] == initial_condition[3])
    @constraint(model, R[1] == initial_condition[4])
    @constraint(model, D[1] == initial_condition[5])

    @constraint(model, [t = 2:(T+1)], 0 ≤ S[t] ≤ 1)
    @constraint(model, [t = 2:(T+1)], 0 ≤ E[t] ≤ 1)
    @constraint(model, [t = 2:(T+1)], 0 ≤ I[t] ≤ I_max)
    @constraint(model, [t = 2:(T+1)], 0 ≤ R[t] ≤ 1)
    @constraint(model, [t = 2:(T+1)], 0 ≤ D[t] ≤ D_max)

    # we constrain our policy to lie between 0 and η_max and define
    # the integral of the intervention to be equal to η_total
    # assuming that the intervention is piecewise constant during each
    # T
    @constraint(model, [t = 1:(T+1)], 0 ≤ η[t] ≤ η_max)
    @constraint(model, sum(η) == η_total)

    # to simplify the model constraint, we define nonlinear expressions for
    # the various states. We only need a vector that is T long
    @NLexpression(
        model,
        exposed[t=1:T],
        (1 - exp(-(1 - η[t]) * parameters[1] * parameters[2] * I[t] * δt)) * S[t]
    )
    @NLexpression(
        model,
        infected[t=1:T],
        (1 - exp(-parameters[3] * δt)) * E[t]
    )
    @NLexpression(
        model,
        recovery[t=1:T],
        (1 - exp(-parameters[2] * δt)) * I[t])
    @NLexpression(
        model,
        death[t=1:T],
        (1 - exp(-parameters[5] * parameters[2] * δt) * I[t])
    )

    # we add additional constraints corresponding to the function map for S, E, I, R and D.
    # These have to be nonlinear constraints due to the inclusion of nonlinear expressions
    @NLconstraint(model, [t = 1:T], S[t+1] == S[t] - exposed[t])
    @NLconstraint(model, [t = 1:T], E[t+1] == E[t] + exposed[t] - infected[t])
    @NLconstraint(model, [t = 1:T], I[t+1] == I[t] + infected[t] - recovery[t])
    @NLconstraint(model, [t = 1:T], R[t+1] == R[t] + recovery[t] - death[t])
    @NLconstraint(model, [t = 1:T], D[t+1] == D[t] + death[t])

    # we declare ourr objective as minimizing the total cost of the intervention  plus smoothing penalty
    @objective(model, Min, η_total)
    silent ? set_silent(model) : nothing
    optimize!(model)
    @info termination_status(model)

    η_opt = value.(η)
    # ritorno il vettore di quando applicare le contromisure
    ηt = unique(map((x) -> trunc(Int, x), findall(x -> x > 1e-3, η_opt) * δt))
    filter!(e -> e ≠ 0, ηt)
    # ritorno il vettore del valore delle contromisure utilizzate in
    # durante il periodo specifico
    η_opt_t = η_opt[trunc(Int, ηt[1] / δt):trunc(Int, 1 / δt):trunc(Int, ηt[end] / δt)]
    plt = nothing
    if showplot
        ts = collect(0:δt:timeframe[2])
        plt = plot(ts, value.(S), label="S", xlabel="Time", ylabel="Number")
        plot!(ts, value.(E), label="E")
        plot!(ts, value.(I), label="I")
        plot!(ts, value.(R), label="R")
        plot!(ts, value.(D), label="D")
        plot!(ts, value.(η), label="Optimized η")
        hline!([I_max], color=:gray, linestyle=:dashdotdot, alpha=0.5, label="Threshold I")
        hline!([η_max], color=:orange, linestyle=:dashdotdot, alpha=0.5, label="Threshold η")
    end
    return ηt, η_opt_t, plt
end

ηt, η_opt_t, plt = controller!(
    [0.99, 0, 0.01, 0, 0],
    [3.54, 1 / 14, 1 / 5, 1 / 280, 0.007, 0.0, 0.0];
    showplot=true
)
plt
