# JEPMC (Julia EPidemiology Model and Control)

A library to explore epidemic models and their controls.

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)


# How to use the library

First of all be sure to import the package correctly running 
```julia
julia> import Pkg; Pkg.add("JPEMC")
```

After correctly importing the library you should be able to use it
inside your own project as follow
```julia
using JEPMC
```

## Let's look at a simple example

Let's say we want to simulate the behaviour of a specific epidemiology
phenomena like COVID-19. First of all it's good to know how the
simulation works. The simulation works try to mimic a *Social Network
Graph*, with each node represents a *Point of Interest (PoI)* where
the simulation of the epidemic take place as a system of *Ordinary
Differentiable Equation (ODE)*.

Now we are ready to define our model. We roughly need to know the
following set of parameters:


```julia
params = Dict(
    :numNodes => ..., # number of total nodes that make the
                      # graph. Default 50
    :edgesCoverage => ..., # try to generalize the how many edges
                           # there are between the nodes. Generally
                           # more edges mean a flux of migration more
                           # wide.
                           # If not specified is assumed :high.
                           # The possible values are :high, :medium
                           # and :low 
    :initialNodeInfected => ..., # number of initial node from where
                                 # the epidemic start. 
                                 # Default 1
    :param => ..., # a vector that identify the parameters of an
                   # epidemic.
                   # Typically represents the following variables:
                   # R₀, γ = infectivity period, σ = exposed period,
                   # ω = immunity period, δ = mortality rate.
                   # Default [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
    :avgPopulation => ..., # average population from wich the
                           # population of each node is created
                           # following an exponential distribution.
                           # Default 10_000
    :maxTravelingRate => ..., # maximum flux of people from one node
                              # to another.
                              # This value is used to create a matrix
                              # of migration between nodes.
                              # Default 0.001 
    :control => ..., # boolean value that notify the use of a
                     # non-pharmaceutical control.
                     # Default false
    :vaccine => ..., # boolean value that notify the simulation of a 
                     # random research and than application of a
                     # pharmaceutical control.
                     # Default false
    :seed => ..., # For reproducibility.
                  # Default 1234
)
```

Now that we know the set of parameters useful to run the simulation,
we create our first model as follow:

```julia
model = JEPMC.init(; numNodes = 8,
                     edgesCoverage = :low,
                     avgPopulation = 1000)
julia> StandardABM with 8 agents of type Node
 space: periodic continuous space with (100.0, 100.0) extent and spacing=4.0
 scheduler: fastest
 properties: graph, param, step, control, numNodes, migrationMatrix, integrator, vaccine

model.migrationMatrix
julia> 8×8 SparseArrays.SparseMatrixCSC{Float64, Int64} with 32 stored entries:
  ⋅          2.66504e-5   ⋅            ⋅           ⋅          3.35803e-5    ⋅           1.27017e-5
 9.69437e-6   ⋅          1.45804e-5   7.31614e-6  1.34918e-5   ⋅            ⋅            ⋅
  ⋅          1.38723e-5   ⋅           7.30609e-6  1.31818e-5  1.62707e-5   3.46623e-5    ⋅
  ⋅          0.00025027  0.000262684   ⋅           ⋅          0.000336501   ⋅           7.66992e-5
  ⋅          1.50257e-5  1.54299e-5    ⋅           ⋅          1.78331e-5   3.93614e-5   9.37485e-6
 9.01767e-6   ⋅          1.26247e-5   7.26198e-6  1.1821e-5    ⋅            ⋅            ⋅
  ⋅           ⋅          8.94326e-6    ⋅          8.67602e-6   ⋅            ⋅           7.60861e-6
 1.61393e-5   ⋅           ⋅           7.83197e-6  2.94039e-5   ⋅           0.000108267   ⋅
```

The property *integrator* of the model is created once the model is
instantiated and is used to create an array of **ODEProblem** storing
the relative ODE system of each node. Here the initial parameter
*:param* is modified adding two extra values η, ξ corresponding to the
strictness of the non-pharmaceutical countermeasures applied in a
specific node and the vaccine coverage when a vaccine is found. The η
value try to summarize very roughly the contermeasures associated to
the [OxCGRT project](https://github.com/OxCGRT/covid-policy-tracker).

This additional parameters cannot be modified for now but could be in
a future update.

Once the model is being instantiated, it's time to make it run and
collect the output. The function that is responsible to that takes as
input a bunch of parameters but only a few is really important and
useful

```julia
data = JEPMC.collect!(
    model; # the model we want to simulate
    n = ..., # the amount of steps we want our models to do
)
```
The result will be an array of *DataFrames* encoding the evolution of each of the nodes in the graph
```julia
data = JEPMC.collect!(model; n=300)
julia> 8-element Vector{DataFrame}:
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      1  [1.0, 0.0, 0.0, 0.0, 0.0]           0.728543      0.0     ⋯
   2 │     1      1  [1.0, 0.0, 0.0, 0.0, 0.0]           0.728543      0.0
   3 │     2      1  [1.0, 2.81622e-13, 6.55914e-13, …   0.728543      0.0
   4 │     3      1  [1.0, 1.14603e-12, 2.0062e-12, 4…   0.728543      0.0
   5 │     4      1  [1.0, 2.81526e-12, 4.19061e-12, …   0.728543      0.0     ⋯
   6 │     5      1  [1.0, 5.49071e-12, 7.41765e-12, …   0.728543      0.0
   7 │     6      1  [1.0, 9.38675e-12, 1.19501e-11, …   0.728543      0.0
   8 │     7      1  [1.0, 1.47507e-11, 1.81014e-11, …   0.728543      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      1  [0.182701, 0.00399672, 0.015527,…   1.0           0.0     ⋯
 296 │   295      1  [0.184841, 0.00391685, 0.0150742…   1.0           0.0
 297 │   296      1  [0.186988, 0.0038401, 0.0146417,…   1.0           0.0
 298 │   297      1  [0.189139, 0.00376634, 0.0142285…   1.0           0.0
 299 │   298      1  [0.191296, 0.00369545, 0.0138336…   1.0           0.0     ⋯
 300 │   299      1  [0.193458, 0.00362733, 0.0134562…   1.0           0.0
 301 │   300      1  [0.195623, 0.00356185, 0.0130954…   1.0           0.0
                                                  2 columns and 286 rows omitted
...
```
The last thing we want to do is plot our data
```julia
plt = JEPMC.plot_model(data)
```
![Plot Without Intervention](https://github.com/DrStiev/JEPMC/blob/main/readmeimg/plot.svg?raw=true)


## Now let's try to activate some type of control

Let's create a model in which the controller is active as a
non-pharmaceutical tipe of control, like mask, smart working, social
distancing etc...

```julia
options=Dict(
    :tolerance => ..., # Minimum threshold of infected individuals 
                       # to call the controller
	:dt => ..., # Timestep used to update the controller
                # countermeasures
	:step => ..., # Integration step for the ODE resolutor
	:maxiters => ..., # Maximum number of iterations for the neural
                      # network controller
)

model = JEPMC.init(; 
    numNodes = 8,
    edgesCoverage = :low, 
    avgPopulation = 1000,
    control = true, 
    control_options = options
)
data = JEPMC.collect!(model; n = 300)
plt = JEPMC.plot_model(data)
```

![Non-Pharmaceutical Countermeasures Plot](https://github.com/DrStiev/JEPMC/blob/main/readmeimg/controlPlot.svg?raw=true)

The controller function is not currently available to be modified from
the user, but it could be in the future. The reason for now is because
mainly the controller function is modeled to be flexible on a general
type of compartmental model, in this case the SEIR(S) one, that it's
not intended to be externally modified.

