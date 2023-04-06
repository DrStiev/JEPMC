using Pkg
Pkg.activate(".")
Pkg.instantiate()
# Pkg.precompile()
# Pkg.resolve()

module test
    @time include("params.jl") # ottengo i parametri che passo al modello
    @time include("graph.jl") # creo il modello ad agente
    @time include("continuous.jl")
    @time include("ode.jl") # creo un modello ode che fa da supporto al modello ad agente 
    @time include("optimizer.jl") # trovo i parametri piu' adatti al modello cercando di minimizzare specifici parametri
    @time include("controller.jl") # applico tecniche di ML per addestrare un modello e estrapolare policy di gestione

    # test parameters creation
    @time u0, tspan, p = model_params.extract_data_from_csv_ode("csv_files/data.csv")
    @time params = model_params.dummyparams()
    @time c_params = model_params.c_dummyparams()

    # FIXME: death count troppo elevato
    # test ODE solver
    @time prob = ode.get_ODE_problem(ode.SEIRD!, u0, tspan, p)
    @time sol = ode.get_solution(prob)
    @time ode.line_plot(sol)

    #FIXME
    # test graphspace abm
    @time model = graph.init(; params...)
    @time data = graph.collect(model)
    @time graph.line_plot(data)

    # test continuousspace abm
    @time c_model = continuous.model_init(; c_params...)
    @time c_data = continuous.collect(c_model)
    @time continuous.line_plot(c_data)
end