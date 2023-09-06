# Julia Epidemiology Model and Control (JEPMC)

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

JEPMC is a library designed for the exploration of epidemic models and their corresponding control mechanisms.

## Overview of the Library

The primary focus of this library is to combine the capabilities and flexibility of an Agent-Based simulation using the *Agents.jl* framework with the stability and performance offered by classical mathematical models defined through systems of Ordinary Differential Equations (ODEs). Additionally, JEPMC harnesses the power of Neural Networks to extrapolate and learn the underlying mechanisms essential for epidemic control.

## Epidemic Model Description

The model implemented in JEPMC is relatively straightforward and emulates a social graph structure. This approach is particularly valuable for simulation purposes. The central concept involves modeling a network of Points of Interest (PoI), where each PoI simulates a node within the graph. These nodes are interconnected to varying degrees. Each node has its own system of defining ODEs, and during each simulation step, one integration step corresponds to the ODE system's evolution.

This approach may initially seem overly deterministic and lacking the stochastic behavior typically associated with simulations that produce emergent behavior. However, the way the model is formulated introduces a touch of stochasticity that proves beneficial, as discussed further below.

Finally, the model incorporates a NeuralODE, responsible for identifying and learning the appropriate countermeasures to apply in dynamic situations that change over time. A dedicated section will delve deeper into this aspect.

## How to Utilize the Library

Begin by importing the package correctly with the following command:

```julia
julia> import Pkg; Pkg.add("JPEMC")
```

Once the library is imported, you can use it in your project as follows:

```julia
using JEPMC
```

## A Simple Example

Suppose you wish to simulate the behavior of a specific epidemiological phenomenon, such as COVID-19. First, you need to define a set of parameters roughly as follows:

```julia
params = Dict(
    :numNodes => ..., # Number of total nodes in the graph (default: 50)
    :edgesCoverage => ..., # Degree of interconnection between nodes (default: :high)
    :initialNodeInfected => ..., # Initial number of nodes where the epidemic starts (default: 1)
    :param => ..., # Parameters representing epidemic characteristics (default: [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01])
    :avgPopulation => ..., # Average population from which individual node populations are generated (default: 10,000)
    :maxTravelingRate => ..., # Maximum rate of people traveling from one node to another (default: 0.001)
    :control => ..., # Boolean flag indicating the use of non-pharmaceutical control measures (default: false)
    :vaccine => ..., # Boolean flag indicating simulation of pharmaceutical control measures (default: false)
    :seed => ..., # For reproducibility (default: 1234)
)
```

With these parameters defined, you can create your initial model as follows:

```julia
model = JEPMC.init(;
    numNodes = 8,
    avgPopulation = 1000,
    edgesCoverage = :high,
    seed = 42
)
```

This command initializes your model with the specified parameters. Notably, the `migrationMatrix` property represents the coverage of edges between nodes in sparse matrix form, providing information on the existence and magnitude of population flow between nodes. Additionally, the `integrator` property creates an array of ODEProblem instances, each corresponding to a node's ODE system. The `param` parameter is modified to include two extra values, η and ξ, which signify the strictness of non-pharmaceutical countermeasures and vaccine coverage when applicable.

Once the model is instantiated, you can run it to collect output data using the `collect!` function:

```julia
data = JEPMC.collect!(
    model,
    n = 300
)
```

The result is an array of DataFrames encoding the evolution of each node in the graph, including information about their status, happiness, and more.

## Model Operation

Without external intervention from a controller (whether pharmaceutical or non-pharmaceutical), the simulation and model operation follow these basic steps:

1. Using the migrationMatrix, each agent, represented as a system of ODEs, calculates the new proportions of its status vector, updating the percentages representing individuals transitioning from one node to another.
2. The integrator is notified of potential changes in parameter status, prompting an update.
3. The model advances by calculating the new status of each agent, progressing by one integration step with the integrator.

You can visualize the results of your simulation using the following command:

```julia
plt = JEPMC.plot_model(data)
```

![Plot Without Intervention](https://github.com/DrStiev/JEPMC/blob/main/readmeimg/plot.png?raw=true)

This command generates graphical representations of the model's behavior, illustrating the epidemiological trends of each node within the graph.

## Introducing a Controller

You can introduce control mechanisms to the model, allowing it to autonomously adjust using a NeuralODE controller. The controller's functioning can be summarized as follows:

1. Given a snapshot of a node and additional parameters, the controller is instantiated, creating a Neural Network (NN) via the Lux framework.
2. The NN is integrated as an estimator within the known epidemic model, and an additional equation accounts for the relationship between countermeasures, the environment, and happiness.
3. The initial model is ready to enter the training loop, with the results used as values in the simulation model.

When implementing the controller, you have several options:

```julia
control_options = Dict(
    :tolerance => 1e-3, # Minimum threshold of infected individuals before controller activation (default: 1e-3)
    :dt => 10, # Time step for controller countermeasure updates (default: 10)
    :step => 3, # Integration step for the ODE solver (default: 3)
    :maxiters => 100, # Maximum number of iterations for the neural network training loop (default: 100)
    :loss => missing, # Custom loss function for the neural network (default: missing)
    :υ_max => missing # Custom attention threshold used as an upper limit for controller countermeasures (default: missing)
)
```

The `tolerance` parameter determines when the controller becomes alert to even small changes in population health. The `dt` parameter simulates the time between countermeasures and their research and validation. You can specify a custom `loss` function for the neural network training, although it requires expertise in the NN structure. The `υ_max` function serves as an alert meter for the population, usually based on a Cumulative Distribution Function (CDF) of a Beta distribution.

Incorporating the controller into your model can be done as follows:

```julia
model = JEPMC.init(; 
    numNodes = 8,
    edgesCoverage = :high, 
    avgPopulation = 1000,
    control = true, 
    control_options = control_options
)
data = JEPMC.collect!(model; n = 300)
plt = JEPMC.plot_model(data)
```

![Non-Pharmaceutical Countermeasures Plot](https://github.com/DrStiev/JEPMC/blob/main/readmeimg/controlPlot.png?raw=true)

This modification shifts the behavior of the model in time, slowing the spread of the pandemic due to the application of non-pharmaceutical countermeasures. Notably, curves related to the force of infection (FoI) exhibit shallower troughs and lower peaks, signifying the effectiveness of the countermeasures.

Please note that interpreting the results of the controller's actions may require human intervention, as the learned countermeasures are expressed as cumulative values rather than specific instructions (e.g., mask mandates or lockdowns).

