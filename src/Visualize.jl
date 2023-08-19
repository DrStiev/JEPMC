# TODO: add InteractiveDynamics to simulate visually the model
# TODO: inspired by https://github.com/DrStiev/julia-epidemiology-models/blob/6f18493dd75b5632b7cb02358d83fb2d7f6e2442/src/pplot.jl
# TODO: use already defined function as plot_system_graph as reference
# TODO: slides

using Plots, Agents, Dates
include("SocialNetworkABM.jl")
include("ABMUtils.jl")

# very inefficient
for i in 1:400
    model = init(; numNodes=10)
    data = collect!(model; n=i, showprogress=false)
    plt = plot(plot_system_graph(model), plot_model(data))
    display(plt)
end

# TODO: https://juliadynamics.github.io/Agents.jl/stable/examples/agents_visualizations/#Agents.abmvideo
