using Pkg
Pkg.activate(".")
Pkg.instantiate()
# Pkg.precompile()
# Pkg.resolve()

@time include("params.jl") # ottengo i parametri che passo al modello
@time include("graph.jl") # creo il modello ad agente
@time include("ode.jl") # creo un modello ode che fa da supporto al modello ad agente 
@time include("optimizer.jl") # trovo i parametri piu' adatti al modello cercando di minimizzare specifici parametri
@time include("controller.jl") # applico tecniche di ML per addestrare un modello e estrapolare policy di gestione
@time include("plot.jl") # plotto i risultati