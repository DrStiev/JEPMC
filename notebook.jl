### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 59935147-8e82-4196-aa5b-dc6d0739eccb
import Pkg; Pkg.activate(".")#Pkg.add("JEPMC")

# ╔═╡ 3352e19d-983d-41fa-933c-92f53c5445a9
using PlutoUI

# ╔═╡ 4f24027b-6571-46d3-bd74-a2feb9e2b264
# using JEPMC
include("src/JEPMC.jl")

# ╔═╡ 111f56e9-c901-4606-8e5f-26663e6353c2
md"""
# Julia Epidemiology Model and Control (JEPMC)

JEPMC is a library designed for the exploration of epidemic models and their corresponding control mechanisms.

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
"""

# ╔═╡ f3b43cb5-6859-4ada-b516-31416287c3bd
md"""
## Overview of the Library

The primary focus of this library is to combine the capabilities and flexibility of an Agent-Based simulation using the *Agents.jl* framework with the stability and performance offered by classical mathematical models defined through systems of Ordinary Differential Equations (ODEs). Additionally, JEPMC harnesses the power of Neural Networks to extrapolate and learn the underlying mechanisms essential for epidemic control.

## Epidemic Model Description

The model implemented in JEPMC is relatively straightforward and emulates a social graph structure. This approach is particularly valuable for simulation purposes. The central concept involves modeling a network of Points of Interest (PoI), where each PoI simulates a node within the graph. These nodes are interconnected to varying degrees. Each node has its own system of defining ODEs, and during each simulation step, one integration step corresponds to the ODE system's evolution.

This approach may initially seem overly deterministic and lacking the stochastic behavior typically associated with simulations that produce emergent behavior. However, the way the model is formulated introduces a touch of stochasticity that proves beneficial, as discussed further below.

Finally, the model incorporates a NeuralODE, responsible for identifying and learning the appropriate countermeasures to apply in dynamic situations that change over time. A dedicated section will delve deeper into this aspect.

## How to Utilize the Library

Begin by importing the package correctly with the following command:
"""

# ╔═╡ a31858a1-1ccd-45f0-ac36-12912fa74fa8
md"""
Once the library is imported, you can use it in your project as follows:
"""

# ╔═╡ 1013d06d-0b44-4e35-bd5b-f99a35e55ffe
md"""
## A Simple Example

Suppose you wish to simulate the behavior of a specific epidemiological phenomenon, such as COVID-19. First, you need to define a set of parameters roughly as follows:
"""

# ╔═╡ 94fd6fbd-6fa3-4801-8339-3ab9b33ba7a0
params = Dict(
    :numNodes => 50, # Number of total nodes in the graph (default: 50)
    :edgesCoverage => :high, # Degree of interconnection between nodes (default: :high)
    :initialNodeInfected => 1, # Initial number of nodes where the epidemic starts (default: 1)
    :param => [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01], # Parameters representing epidemic characteristics (default: [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01])
    :avgPopulation => 10_000, # Average population from which individual node populations are generated (default: 10,000)
    :maxTravelingRate => 0.001, # Maximum rate of people traveling from one node to another (default: 0.001)
    :control => false, # Boolean flag indicating the use of non-pharmaceutical control measures (default: false)
    :vaccine => false, # Boolean flag indicating simulation of pharmaceutical control measures (default: false)
    :seed => 1234, # For reproducibility (default: 1234)
)

# ╔═╡ 0bae9085-c4d8-4eeb-89e7-6ea70620f636
md"""
With these parameters defined, you can create your initial model as follows:
"""

# ╔═╡ f77570f7-6a41-456b-8b6d-c194943fb83d
@bind numNodes Slider(8:8:80, default=8)

# ╔═╡ 2499c056-9b5a-4fd8-b7b1-6cf3b9c576fa
@bind avgPopulation Slider(1000:1000:10_000, default=1000)

# ╔═╡ 5b263e70-733d-4e9d-920b-8071ca52301f
@bind edgesCoverage Select([:high, :medium, :low])

# ╔═╡ 0ec8d3ab-0e2c-47d0-834f-5d74134fc2dd
model = JEPMC.init(;
    numNodes = numNodes,
    avgPopulation = avgPopulation,
    edgesCoverage = edgesCoverage,
    seed = 42
)

# ╔═╡ de089f9c-4508-4e9c-8e9b-c6c3b13e706a
md"""
This command initializes your model with the specified parameters. Notably, the `migrationMatrix` property represents the coverage of edges between nodes in sparse matrix form, providing information on the existence and magnitude of population flow between nodes. Additionally, the `integrator` property creates an array of ODEProblem instances, each corresponding to a node's ODE system. The `param` parameter is modified to include two extra values, η and ξ, which signify the strictness of non-pharmaceutical countermeasures and vaccine coverage when applicable.

Once the model is instantiated, you can run it to collect output data using the `collect!` function:
"""

# ╔═╡ a406e236-d104-4a7d-b919-ed9857ac3970
model.properties

# ╔═╡ f7710138-b78d-4bf9-9cce-5953e73a8183
data = JEPMC.collect!(
    model,
    n = 300
)

# ╔═╡ 874b14a3-aeae-4185-b1af-e9d0c2eb91dd
md"""
The result is an array of DataFrames encoding the evolution of each node in the graph, including information about their status, happiness, and more.

## Model Operation

Without external intervention from a controller (whether pharmaceutical or non-pharmaceutical), the simulation and model operation follow these basic steps:

1. Using the migrationMatrix, each agent, represented as a system of ODEs, calculates the new proportions of its status vector, updating the percentages representing individuals transitioning from one node to another.
2. The integrator is notified of potential changes in parameter status, prompting an update.
3. The model advances by calculating the new status of each agent, progressing by one integration step with the integrator.

You can visualize the results of your simulation using the following command:
"""

# ╔═╡ 174eaad4-8308-418e-87af-1d2855852de6
plt = JEPMC.plot_model(data)

# ╔═╡ eba06c21-e306-4ddb-906f-9fee3043d631
md"""
This command generates graphical representations of the model's behavior, illustrating the epidemiological trends of each node within the graph.

## Introducing a Controller

You can introduce control mechanisms to the model, allowing it to autonomously adjust using a NeuralODE controller. The controller's functioning can be summarized as follows:

1. Given a snapshot of a node and additional parameters, the controller is instantiated, creating a Neural Network (NN) via the Lux framework.
2. The NN is integrated as an estimator within the known epidemic model, and an additional equation accounts for the relationship between countermeasures, the environment, and happiness.
3. The initial model is ready to enter the training loop, with the results used as values in the simulation model.

When implementing the controller, you have several options:
"""

# ╔═╡ 4f3b494e-7dad-47d7-8716-638307c69669
control_options = Dict(
    :tolerance => 1e-3, # Minimum threshold of infected individuals before controller activation (default: 1e-3)
    :dt => 10, # Time step for controller countermeasure updates (default: 10)
    :step => 3, # Integration step for the ODE solver (default: 3)
    :maxiters => 100, # Maximum number of iterations for the neural network training loop (default: 100)
    :loss => missing, # Custom loss function for the neural network (default: missing)
    :υ_max => missing # Custom attention threshold used as an upper limit for controller countermeasures (default: missing)
)

# ╔═╡ 2a16a68f-ecf6-469e-be4f-4d83adeefed8
md"""
The `tolerance` parameter determines when the controller becomes alert to even small changes in population health. The `dt` parameter simulates the time between countermeasures and their research and validation. You can specify a custom `loss` function for the neural network training, although it requires expertise in the NN structure. The `υ_max` function serves as an alert meter for the population, usually based on a Cumulative Distribution Function (CDF) of a Beta distribution.

Incorporating the controller into your model can be done as follows:
"""

# ╔═╡ b87f0638-a6d9-481e-87cf-e4f10a361a54
@bind control Select([true, false])

# ╔═╡ 399b8e88-9203-4db6-b71b-d22a205e5624
model_control = JEPMC.init(;
    numNodes = numNodes,
    edgesCoverage = edgesCoverage,
    avgPopulation = avgPopulation,
    control = control,
    control_options = control_options
)

# ╔═╡ 332c69ff-249c-43b6-95b0-85e508f002c4
data_control = JEPMC.collect!(model_control; n = 300)

# ╔═╡ ed12a389-a6c5-4837-92b3-8bc09822d2d3
plt_control = JEPMC.plot_model(data_control)

# ╔═╡ 2af22f3f-9c91-4a53-aa74-968c05894ef3
md"""
This modification shifts the behavior of the model in time, slowing the spread of the pandemic due to the application of non-pharmaceutical countermeasures. Notably, curves related to the force of infection (FoI) exhibit shallower troughs and lower peaks, signifying the effectiveness of the countermeasures.

Please note that interpreting the results of the controller's actions may require human intervention, as the learned countermeasures are expressed as cumulative values rather than specific instructions (e.g., mask mandates or lockdowns).
"""

# ╔═╡ Cell order:
# ╠═111f56e9-c901-4606-8e5f-26663e6353c2
# ╠═f3b43cb5-6859-4ada-b516-31416287c3bd
# ╠═59935147-8e82-4196-aa5b-dc6d0739eccb
# ╠═3352e19d-983d-41fa-933c-92f53c5445a9
# ╠═a31858a1-1ccd-45f0-ac36-12912fa74fa8
# ╠═4f24027b-6571-46d3-bd74-a2feb9e2b264
# ╠═1013d06d-0b44-4e35-bd5b-f99a35e55ffe
# ╠═94fd6fbd-6fa3-4801-8339-3ab9b33ba7a0
# ╠═0bae9085-c4d8-4eeb-89e7-6ea70620f636
# ╠═f77570f7-6a41-456b-8b6d-c194943fb83d
# ╠═2499c056-9b5a-4fd8-b7b1-6cf3b9c576fa
# ╠═5b263e70-733d-4e9d-920b-8071ca52301f
# ╠═0ec8d3ab-0e2c-47d0-834f-5d74134fc2dd
# ╠═de089f9c-4508-4e9c-8e9b-c6c3b13e706a
# ╠═a406e236-d104-4a7d-b919-ed9857ac3970
# ╠═f7710138-b78d-4bf9-9cce-5953e73a8183
# ╠═874b14a3-aeae-4185-b1af-e9d0c2eb91dd
# ╠═174eaad4-8308-418e-87af-1d2855852de6
# ╠═eba06c21-e306-4ddb-906f-9fee3043d631
# ╠═4f3b494e-7dad-47d7-8716-638307c69669
# ╠═2a16a68f-ecf6-469e-be4f-4d83adeefed8
# ╠═b87f0638-a6d9-481e-87cf-e4f10a361a54
# ╠═399b8e88-9203-4db6-b71b-d22a205e5624
# ╠═332c69ff-249c-43b6-95b0-85e508f002c4
# ╠═ed12a389-a6c5-4837-92b3-8bc09822d2d3
# ╠═2af22f3f-9c91-4a53-aa74-968c05894ef3
