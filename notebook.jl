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

# ╔═╡ 311ab5ea-3ade-4a25-a13d-cec2d96e8b35
using Distributions, Plots

# ╔═╡ 4f24027b-6571-46d3-bd74-a2feb9e2b264
# using JEPMC
include("src/JEPMC.jl")

# ╔═╡ 111f56e9-c901-4606-8e5f-26663e6353c2
md"""
# JEPMC (Julia EPidemiology Model and Control)

A library to explore epidemic models and their controls.

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

"""

# ╔═╡ f3b43cb5-6859-4ada-b516-31416287c3bd
md"""
## What is the library?

The library main focus is to combine the power and flexibility of an Agent-Based simulation using the framework *Agents.jl*, with the stability and performance of the *classical mathematical model* defined with a *system of Ordinary Differentiable Equation (ODE)*. Then it use the power of a *Neural Network* to extrapolate and learning all the interesting mechanism at the base of an epidemic control. 

## The model

The model is relatively simple and mimic a social graph structure. This approach is found to be very useful for the sake of the simulation. The main concept is to model a *network of Point of Interest (PoI)* where each PoI simulate a node of the graph and is more or less connected to all the other nodes. Each node have it's own defining system of ODE and at each step of the simulation correspond one integration step of the system of ODE. 

This approach seems to be extremely deterministic and lacking in one of the major cons of using a simulation, the stochastic behaviour leading to an emergent behaviour, but the way the model is defined introduce a sparkle of stochasticity useful to this approach. This will be discuss futher on. 

Last, the model will call on a NeuralODE that is responsible to find and learn the right countermeasures to apply in a given situation, that dynamically changes over time. This section will be explored in much grater detail in a dedicated section.
"""

# ╔═╡ 14ad3bdb-57ea-4a32-8e28-31df2054fe9a
md"""
## How to use the library

First of all be sure to import the package correctly, running the following command:
"""

# ╔═╡ a31858a1-1ccd-45f0-ac36-12912fa74fa8
md"""
After correctly importing the library you should be able to use it inside your own project as follow:
"""

# ╔═╡ 1013d06d-0b44-4e35-bd5b-f99a35e55ffe
md"""
## Let's look at a simple example

Let's say we want to simulate the behaviour of a specific epidemiology phenomena like COVID-19. First of all we roughly need to know the following set of parameters:
"""

# ╔═╡ 94fd6fbd-6fa3-4801-8339-3ab9b33ba7a0
params = Dict(
	:numNodes => 50, # number of total nodes that make the
                     # graph. Default 50
    :edgesCoverage => :high, # try to generalize the how many edges there are between the nodes. Generally more edges mean a flux of migration more wide. If not specified is assumed :high. The possible values are :high, :medium and :low 
    :initialNodeInfected => 1, # number of initial node from where the epidemic start. Default 1
    :param => [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01], # a vector that identify the parameters of an epidemic. Typically represents the following variables: R₀, γ = infectivity period, σ = exposed period, ω = immunity period, δ = mortality rate. Default [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
    :avgPopulation => 10_000, # average population from wich the population of each node is created following an exponential distribution. Default 10_000
    :maxTravelingRate => 0.001, # maximum flux of people from one node to another. This value is used to create a matrix of migration between nodes. Default 0.001 
    :control => false, # boolean value that notify the use of a non-pharmaceutical control.Default false
    :vaccine => false, # boolean value that notify the simulation of a random research and than application of a pharmaceutical control. Default false
    :seed => 1234, # For reproducibility. Default 1234
)

# ╔═╡ 0bae9085-c4d8-4eeb-89e7-6ea70620f636
md"""
Now that we know the set of parameter useful to run a simulation, we can create our first model
"""

# ╔═╡ f77570f7-6a41-456b-8b6d-c194943fb83d
@bind numNodes Slider(8:8:80, default=8)

# ╔═╡ 2499c056-9b5a-4fd8-b7b1-6cf3b9c576fa
@bind avgPopulation Slider(1000:1000:10_000, default=1000)

# ╔═╡ 5b263e70-733d-4e9d-920b-8071ca52301f
@bind edgesCoverage Select([:high, :medium, :low])

# ╔═╡ 0ec8d3ab-0e2c-47d0-834f-5d74134fc2dd
model = JEPMC.init(;
	numNodes = numNodes, avgPopulation = avgPopulation, edgesCoverage = edgesCoverage,
	seed = 42
)

# ╔═╡ de089f9c-4508-4e9c-8e9b-c6c3b13e706a
md"""
As we see the output of the line of code tells us the basic property of our model. The most interesting properties among all the others are the migrationMatrix property, and the integrator one.

The former is associated to the representation in a *sparse matrix form* of the coverage of all the edges from each node to another. This matrix gives 2 basic information: if an edge exists between two nodes, and if exists what is the amount of individuals that will pass from the source node into the destination node. This amount is a percentage. 

The property integrator of the model is created once the model is instantiated and is used to create an array of ODEProblem storing the relative ODE system of each node. Here the initial parameter :param is modified adding two extra values η, ξ corresponding to the strictness of the non-pharmaceutical countermeasures applied in a specific node and the vaccine coverage when a vaccine is found. The η value try to summarize very roughly the contermeasures associated to the [OxCGRT project](https://github.com/OxCGRT/covid-policy-tracker). This additional parametes cannot be modified at the moment but could be in a future update.
"""

# ╔═╡ a406e236-d104-4a7d-b919-ed9857ac3970
model.properties

# ╔═╡ 14106298-3d14-4be3-9dcc-60ef9deefbbc
md"""
Once the model is being instantiated, it's time to make it run to collect the output. The function that is responsible to that is ```collect!``` and takes as input a bunch of parameters but only a few is really important.
"""

# ╔═╡ f7710138-b78d-4bf9-9cce-5953e73a8183
data = JEPMC.collect!(
	model; # the model we want to simulate
	n = 400 # the amount of steps we want our model to do
)

# ╔═╡ 874b14a3-aeae-4185-b1af-e9d0c2eb91dd
md"""
As we can see the result is an array of *DataFrams* each one encoding the evolution of each node of the graph. Some of this information is useful to be discussed.

- The column 	```status``` encodes the snapshot of the system of ODE at a specific timestamp. Each value of the 5-elements array represents one of the following categories: Susceptible (S), Exposed (E), Infected (I), Recovered (R), Dead (D)
- The column ```happiness``` is a tricky column representation, because it's not encoding the real happiness of the population inside of a specific node given a particular situation. Instead is very loosly an approximator that is used to mantain the controller on check with the countermeasures. We will go deeper in the next sections.
- The column ```υ``` encodes 2 things: when a pharmaceutical countermeasure (typically a vaccine) is found and it's efficacy (vaccine coverage)
- The column ```R₀``` describe the R₀ index of a given node, trying to simulate the behaviour and the birth of new variant.
"""

# ╔═╡ 4abf1f1b-34c2-4f29-ab18-4e52bb61d269
md"""
## How the model works

Without the external intervention of a controllorer of any kind (be pharmaceutical or non), the simulation and the model, after initialize the model and the space as desired by the user, follow this simple steps:
1. Given the migrationMatrix each agent, that is represent as a system of ODE, calculate the new proportion of it's status vector, updating all the percentage of it's status vector (ideally representing individuals) that exit from the node and go in another node. 
2. Notify the integrator that the status of its parameter could be changed, so it needs to be updated.
3. Advance the model calculating for each agent it's new status advancing of 1 step with the integrator.
"""

# ╔═╡ 89f0ed65-e5fc-456d-9b8d-ca1abe4b2340
md"""
We finally can show graphically the results of our first simulation
"""

# ╔═╡ 174eaad4-8308-418e-87af-1d2855852de6
plt = JEPMC.plot_model(data)

# ╔═╡ eba06c21-e306-4ddb-906f-9fee3043d631
md"""
## Let's try something different

Let's try to give the model the ability to control itself with the external use of a *NeuralODE controller*. 

First of all, how the controller works? Simple, given the general behaviour of the simulation in the form of a system of ODE we insert a Neural Network inside to estimate the amount of conuntermeasures that should be used to end with a good result, typically minimizing the number of infected individuals.

The general behaviour of the simulation is a known model so our controller should not worry to learn that, the kind of countermeasures instead is an unknown model and so only that should ber learnt. The countermeasures learnt are not explicit like "start using masks, start generalized lockdown etc...", instead is a cumulative value that represent the average summa of all the possible countermeasures applicable. This means that **it's required** the supervision of a human being to interpret this results. 
"""

# ╔═╡ 94cabb18-e1d0-476a-9fb2-ab05e07bceaf
md"""
So, the controller act as follow: 
1. Given a snapshot of a node and a bounch of additional parameters, the controller is instantiated creating the Neural Network (NN) via the ```Lux``` framework.
2. The NN is inserted as estimator inside the known model of the epidemics (generally a SEIR(S) model), and is add one additional equation encoding the relation through the countermeasures, the environment and the happiness.
3. The initial model is ready to be put in the training loop, and after that the result will be used as value to the simulation model
"""

# ╔═╡ 4f3b494e-7dad-47d7-8716-638307c69669
control_options = Dict(
	:tolerance => 1e-3, # Minimum threshoild of infected individuals before call the controller. Default 1e-3
	:dt => 10, # Timestep used to update the controller countermeasures
	:step => 3, # Integration step for the ODE solver
	:maxiters => 100, # Maximum number of iterations for the neural network training loop
	:loss => missing, # custom loss function passed to the neural network
	:υ_max => missing, # custom attention threshold used as additional upper limit to the controller countermeasures result
)

# ╔═╡ 2a16a68f-ecf6-469e-be4f-4d83adeefed8
md"""
As we can see, the option given to the user to control the controller function is relatively small but sharp. The idea behinf this project is that the user can manipulate the simulation not the controller per se, so it's kinda obvious that the option to it's manipulation will be small. But this will not mean that the user is restricted in their actions.

Generally the most interesting of all are the following options:
- ```tolerance```: generally speaking this parameter will allow the controller to be more alert on even the smallest change in the population health. 
- ```dt```: this will simulate the time passing between one countermeasure and the other, simulate the research and the validation of all the possible option
- ```loss``` and ```υ_max```: these parameters are relatively tricky because need the user to be fairly skilled, otherwise it's a smart choiche to leave as they are. The ```loss``` parameter takes a function that is used as a loss function during the training loop of the neural network. This implies that the user should known very well the structure of the NN otherwise could break all the simulation. ```υ_max``` is a function that simulate the alert meter of a population. Generally speaking, alter this will not directly break up the simulation, but it will certainly lead to wonky and unexpected behaviour
"""

# ╔═╡ 58c20303-fce0-48a8-a75f-f2b8e6685b0f
md"""
The default loss is computed as follow: ```loss(x) = sum(abs2, x[3, :]) / sum(abs2, x[end, :])``` and represent the ratio between the number of infected individual and the happiness of a specific node. This value is then minimized. 

The general happiness of a node is estimated inside the controller as part of the ODE system, and is represented as follow: ```H = -(I + D) + (R * (1 - η)) # dH```
"""

# ╔═╡ f162f2d9-fa66-4ba6-902d-8c8048a64f82
md"""
The ```υ_max``` function is defined as the *Cumulative Distribution Fuction (CDF)* of a *Beta* function with parameters 2 and 5 of the value of I of a specific node: ```Distributions.cdf(Distributions.Beta(2, 5), agent.status[3])```

This kind of function mimic pretty well the growth in interest in an epidemic
"""

# ╔═╡ bcac8fc6-856c-4687-b276-89f0888b3967
plot(Distributions.cdf(Distributions.Beta(2, 5), 0.01:0.01:1)) # with 0 as 0% and 100 as 100% of individual infected

# ╔═╡ b87f0638-a6d9-481e-87cf-e4f10a361a54
@bind control Select([true, false])

# ╔═╡ 0501fe59-32eb-4d52-8528-fb2fb5c92f48
model_control = JEPMC.init(;
	numNodes = numNodes, avgPopulation = avgPopulation, edgesCoverage = edgesCoverage,
	control = control, control_options, seed = 42
)

# ╔═╡ cde0f534-223f-41d5-872c-8b53978a66bf
data_control = JEPMC.collect!(model_control; n = 400)

# ╔═╡ 249453ed-9fe7-49e4-aac7-3a4407ff560d
plt_control = JEPMC.plot_model(data_control)

# ╔═╡ Cell order:
# ╠═111f56e9-c901-4606-8e5f-26663e6353c2
# ╠═f3b43cb5-6859-4ada-b516-31416287c3bd
# ╠═14ad3bdb-57ea-4a32-8e28-31df2054fe9a
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
# ╠═14106298-3d14-4be3-9dcc-60ef9deefbbc
# ╠═f7710138-b78d-4bf9-9cce-5953e73a8183
# ╠═874b14a3-aeae-4185-b1af-e9d0c2eb91dd
# ╠═4abf1f1b-34c2-4f29-ab18-4e52bb61d269
# ╠═89f0ed65-e5fc-456d-9b8d-ca1abe4b2340
# ╠═174eaad4-8308-418e-87af-1d2855852de6
# ╠═eba06c21-e306-4ddb-906f-9fee3043d631
# ╠═94cabb18-e1d0-476a-9fb2-ab05e07bceaf
# ╠═4f3b494e-7dad-47d7-8716-638307c69669
# ╠═2a16a68f-ecf6-469e-be4f-4d83adeefed8
# ╠═58c20303-fce0-48a8-a75f-f2b8e6685b0f
# ╠═f162f2d9-fa66-4ba6-902d-8c8048a64f82
# ╠═311ab5ea-3ade-4a25-a13d-cec2d96e8b35
# ╠═bcac8fc6-856c-4687-b276-89f0888b3967
# ╠═b87f0638-a6d9-481e-87cf-e4f10a361a54
# ╠═0501fe59-32eb-4d52-8528-fb2fb5c92f48
# ╠═cde0f534-223f-41d5-872c-8b53978a66bf
# ╠═249453ed-9fe7-49e4-aac7-3a4407ff560d
