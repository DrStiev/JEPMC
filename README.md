# JEPMC (Julia EPidemiology Model and Control)

A library to explore epidemic models and their controls.

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

## What is the library?

The library main focus is to combine the power and flexibility of an Agent-Based simulation using the framework *Agents.jl*, with the stability and performance of the *classical mathematical model* defined with a *system of Ordinary Differentiable Equation (ODE)*. Then it use the power of a *Neural Network* to extrapolate and learning all the interesting mechanism at the base of an epidemic control. 

## The model

The model is relatively simple and mimic a social graph structure. This approach is found to be very useful for the sake of the simulation. The main concept is to model a *network of Point of Interest (PoI)* where each PoI simulate a node of the graph and is more or less connected to all the other nodes. Each node have it's own defining system of ODE and at each step of the simulation correspond one integration step of the system of ODE. 

This approach seems to be extremely deterministic and lacking in one of the major cons of using a simulation, the stochastic behaviour leading to an emergent behaviour, but the way the model is defined introduce a sparkle of stochasticity useful to this approach. This will be discuss futher on. 

Last, the model will call on a NeuralODE that is responsible to find and learn the right countermeasures to apply in a given situation, that dynamically changes over time. This section will be explored in much grater detail in a dedicated section.

## How to use the library

First of all be sure to import the package correctly, running the following command: 
```julia
julia> import Pkg; Pkg.add("JPEMC")
```

After correctly importing the library you should be able to use it inside your own project as follow:
```julia
using JEPMC
```

## Let's look at a simple example

Let's say we want to simulate the behaviour of a specific epidemiology phenomena like COVID-19. First of all we roughly need to know the following set of parameters:

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

Now that we know the set of parameter useful to run a simulation, we can create our first model

```julia
model = JEPMC.init(;
	numNodes = 8, 
    avgPopulation = 1000, 
    edgesCoverage = :high,
    seed = 42
)
julia> StandardABM with 8 agents of type Node
 space: periodic continuous space with (100.0, 100.0) extent and spacing=4.0
 scheduler: fastest
 properties: graph, control_options, param, step, control, numNodes, migrationMatrix, integrator, vaccine
```
As we see the output of the line of code tells us the basic property of our model. The most interesting properties among all the others are the migrationMatrix property, and the integrator one.

The former is associated to the representation in a *sparse matrix form* of the coverage of all the edges from each node to another. This matrix gives 2 basic information: if an edge exists between two nodes, and if exists what is the amount of individuals that will pass from the source node into the destination node. This amount is a percentage. 

The property integrator of the model is created once the model is instantiated and is used to create an array of ODEProblem storing the relative ODE system of each node. Here the initial parameter :param is modified adding two extra values η, ξ corresponding to the strictness of the non-pharmaceutical countermeasures applied in a specific node and the vaccine coverage when a vaccine is found. The η value try to summarize very roughly the contermeasures associated to the [OxCGRT project](https://github.com/OxCGRT/covid-policy-tracker). This additional parametes cannot be modified at the moment but could be in a future update.

Once the model is being instantiated, it's time to make it run to collect the output. The function that is responsible to that is ```collect!``` and takes as input a bunch of parameters but only a few is really important.

```julia
data = JEPMC.collect!(
    model; # the model we want to simulate
    n = 300, # the amount of steps we want our models to do
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
As we can see the result is an array of *DataFrams* each one encoding the evolution of each node of the graph. Some of this information is useful to be discussed.

- The column 	```status``` encodes the snapshot of the system of ODE at a specific timestamp. Each value of the 5-elements array represents one of the following categories: Susceptible (S), Exposed (E), Infected (I), Recovered (R), Dead (D)
- The column ```happiness``` is a tricky column representation, because it's not encoding the real happiness of the population inside of a specific node given a particular situation. Instead is very loosly an approximator that is used to mantain the controller on check with the countermeasures. We will go deeper in the next sections.
- The column ```υ``` encodes 2 things: when a pharmaceutical countermeasure (typically a vaccine) is found and it's efficacy (vaccine coverage)
- The column ```R₀``` describe the R₀ index of a given node, trying to simulate the behaviour and the birth of new variant.

## How the model works

Without the external intervention of a controllorer of any kind (be pharmaceutical or non), the simulation and the model, after initialize the model and the space as desired by the user, follow this simple steps:
1. Given the migrationMatrix each agent, that is represent as a system of ODE, calculate the new proportion of it's status vector, updating all the percentage of it's status vector (ideally representing individuals) that exit from the node and go in another node. 
2. Notify the integrator that the status of its parameter could be changed, so it needs to be updated.
3. Advance the model calculating for each agent it's new status advancing of 1 step with the integrator.

We finally can show graphically the results of our first simulation
```julia
plt = JEPMC.plot_model(data)
```
![Plot Without Intervention](https://github.com/DrStiev/JEPMC/blob/main/readmeimg/plot.svg?raw=true)

The curves showed in the graph represents the behaviour of each node of the model graph with the corresponding epidemiology trend. Sometimes these curves are more accentuated, this case represent a specific trend that is more common across the entire model.

The more accentuated curves in foreground represents the average trend of the entire model. The ribbon style confidence interval of the other two plots represents the confidence interval (CI) of the curve. However this CI, due to the type of plot used to represent the data are prone to show sometimes an incorrect behaviour of the curves. Sometimes when the data are not all equal and are very close to the upper (or lower) limit, this will be ignored to show a CI higher or lower than the maximum (or minimum) possible value.

This behaviour is perfectly shown in the plot below.

## Let's try something different

Let's try to give the model the ability to control itself with the external use of a *NeuralODE controller*. 

First of all, how the controller works? Simple, given the general behaviour of the simulation in the form of a system of ODE we insert a Neural Network inside to estimate the amount of conuntermeasures that should be used to end with a good result, typically minimizing the number of infected individuals.

The general behaviour of the simulation is a known model so our controller should not worry to learn that, the kind of countermeasures instead is an unknown model and so only that should ber learnt. The countermeasures learnt are not explicit like "start using masks, start generalized lockdown etc...", instead is a cumulative value that represent the average summa of all the possible countermeasures applicable. This means that **it's required** the supervision of a human being to interpret this results. 


So, the controller act as follow: 
1. Given a snapshot of a node and a bounch of additional parameters, the controller is instantiated creating the Neural Network (NN) via the ```Lux``` framework.
2. The NN is inserted as estimator inside the known model of the epidemics (generally a SEIR(S) model), and is add one additional equation encoding the relation through the countermeasures, the environment and the happiness.
3. The initial model is ready to be put in the training loop, and after that the result will be used as value to the simulation model

```julia
control_options = Dict(
	:tolerance => 1e-3, # Minimum threshoild of infected 
                        # individuals before call the controller. 
                        # Default 1e-3
	:dt => 10, # Timestep used to update the controller countermeasures
	:step => 3, # Integration step for the ODE solver
	:maxiters => 100, # Maximum number of iterations for the neural network 
                      # training loop
	:loss => missing, # custom loss function passed to the neural network
	:υ_max => missing, # custom attention threshold used as additional 
                       # upper limit to the controller countermeasures result
)
```

As we can see, the option given to the user to control the controller function is relatively small but sharp. The idea behinf this project is that the user can manipulate the simulation not the controller per se, so it's kinda obvious that the option to it's manipulation will be small. But this will not mean that the user is restricted in their actions.

Generally the most interesting of all are the following options:
- ```tolerance```: generally speaking this parameter will allow the controller to be more alert on even the smallest change in the population health. 
- ```dt```: this will simulate the time passing between one countermeasure and the other, simulate the research and the validation of all the possible option
- ```loss``` and ```υ_max```: these parameters are relatively tricky because need the user to be fairly skilled, otherwise it's a smart choiche to leave as they are. The ```loss``` parameter takes a function that is used as a loss function during the training loop of the neural network. This implies that the user should known very well the structure of the NN otherwise could break all the simulation. ```υ_max``` is a function that simulate the alert meter of a population. Generally speaking, alter this will not directly break up the simulation, but it will certainly lead to wonky and unexpected behaviour

The default loss is computed as follow: ```loss(x) = sum(abs2, x[3, :]) / sum(abs2, x[end, :])``` and represent the ratio between the number of infected individual and the happiness of a specific node. This value is then minimized. 

The general happiness of a node is estimated inside the controller as part of the ODE system, and is represented as follow: ```H = -(I + D) + (R * (1 - η)) # dH```

The ```υ_max``` function is defined as the *Cumulative Distribution Fuction (CDF)* of a *Beta* function with parameters 2 and 5 of the value of I of a specific node: ```Distributions.cdf(Distributions.Beta(2, 5), agent.status[3])```

This kind of function mimic pretty well the growth in interest in an epidemic

```julia
model = JEPMC.init(; 
    numNodes = 8,
    edgesCoverage = :low, 
    avgPopulation = 1000,
    control = true, 
    control_options = control_options
)
data = JEPMC.collect!(model; n = 300)
plt = JEPMC.plot_model(data)
```

![Non-Pharmaceutical Countermeasures Plot](https://github.com/DrStiev/JEPMC/blob/main/readmeimg/controlPlot.svg?raw=true)

We can see from the plot that the general behaviour of the model is shift ahead in time, slowing the spreading of the pandemic due to the use of non-pharmaceutical countermeasures. In addition, all the curves related directly to the force of infectin (FoI) have a less deeper pit and less higher peak; in particular the S-Curve do not drop far below the 25%, and the I-Curve do not peak higher as before without the countermeasures.

