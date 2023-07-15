using JuMP, Ipopt, Plots, InfiniteOpt
using Statistics: mean
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_ftc_jump/function_map_ftc_jump.md
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/function_map_vaccine_jump/function_map_vaccine_jump.md
# questo esempio potrebbe essere buono per la NeuralODE
# TODO: https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_lockdown_optimization/ode_lockdown_optimization.md

# https://github.com/epirecipes/sir-julia
function controller!(
    initial_condition::Vector{Float64},
    parameters::Vector{Float64};
    C₀::Float64=0.0,
    υ_max::Float64=0.5, # υ_max e υ_total sono inversamente proporzionali
    υ_total::Float64=10.0,
    timeframe::Tuple{Float64,Float64}=(0.0, 100.0), # anche il tempo potrebbe essere importante
    δt::Float64=0.1,
    showplot::Bool=false
)
    extra_ts = collect(δt:δt:timeframe[2]-δt)
    model = InfiniteModel(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    @infinite_parameter(model, t ∈ [timeframe[1], timeframe[2]], num_supports = length(extra_ts) + 2,
        derivative_method = OrthogonalCollocation(2))
    add_supports(t, extra_ts)

    @variable(model, S ≥ 0, Infinite(t))
    @variable(model, E ≥ 0, Infinite(t))
    @variable(model, I ≥ 0, Infinite(t))
    @variable(model, R ≥ 0, Infinite(t))
    @variable(model, D ≥ 0, Infinite(t))
    @variable(model, C ≥ 0, Infinite(t))

    @variable(model, 0 ≤ υ ≤ υ_max, Infinite(t), start = 0.0)
    @constraint(model, υ_total_constr, ∫(υ, t) ≤ υ_total)

    @objective(model, Min, C(tf))

    @constraint(model, S(0) == initial_condition[1])
    @constraint(model, E(0) == initial_condition[2])
    @constraint(model, I(0) == initial_condition[3])
    @constraint(model, R(0) == initial_condition[4])
    @constraint(model, D(0) == initial_condition[5])
    @constraint(model, C(0) == C₀)

    @constraint(model, S_constr, ∂(S, t) == -(1 - υ) * parameters[1] * parameters[2] * S * I + parameters[4] * R - parameters[7] * S)
    @constraint(model, E_constr, ∂(E, t) == (1 - υ) * parameters[1] * parameters[2] * S * I - parameters[3] * E)
    @constraint(model, I_constr, ∂(I, t) == parameters[3] * E - parameters[2] * I - parameters[5] * I)
    @constraint(model, R_constr, ∂(R, t) == (1 - parameters[5]) * parameters[2] * I - parameters[4] * R + parameters[7] * S)
    @constraint(model, D_constr, ∂(D, t) == parameters[5] * parameters[2] * I)
    @constraint(model, C_constr, ∂(C, t) == parameters[3] * E)

    optimize!(model)
    @info termination_status(model)

    S_opt = value(S, ndarray=true)
    E_opt = value(E, ndarray=true)
    I_opt = value(I, ndarray=true)
    R_opt = value(R, ndarray=true)
    D_opt = value(D, ndarray=true)
    C_opt = value(C, ndarray=true)
    υ_opt = value(υ, ndarray=true)
    obj_opt = objective_value(model)
    ts = value(t)

    # ritorno il vettore di quando applicare le contromisure
    υt = unique(map((x) -> trunc(Int, x), findall(x -> x > 1e-3, υ_opt) * 0.1))
    filter!(e -> e ≠ 0, υt)
    # ritorno il vettore del valore delle contromisure utilizzate durante il periodo specifico
    υ_opt_t = υ_opt[trunc(Int, υt[1] / δt):trunc(Int, 1 / δt):trunc(Int, υt[end] / δt)]
    mean(υ_opt_t)

    plt = nothing
    if showplot
        plt = plot(ts, S_opt, label="S", xlabel="Time", ylabel="Number")
        plot!(ts, E_opt, label="E")
        plot!(ts, I_opt, label="I")
        plot!(ts, R_opt, label="R")
        plot!(ts, D_opt, label="D")
        plot!(ts, C_opt, label="C")
        plot!(ts, υ_opt, label="Optimized υ")
    end
    return (mean(υ_opt_t), plt)
end
