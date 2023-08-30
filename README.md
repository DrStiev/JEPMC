# JEPMC (Julia EPidemiology Model and Control)

A library to explore epidemic models and their controls.

[![Build Status](https://github.com/DrStiev/CovidSim.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DrStiev/CovidSim.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

https://github.com/SciML/SciMLStyle

# How to use the library
First of all be sure to import the package correctly running 
```julia
julia> import Pkg; Pkg.add("JPEMC")
```

After correctly importing the library you should be able to use it inside your own project as follow
```julia
using JEPMC
```

## Let's look at a simple tutorial
Let's say we want to simulate the behaviour of a specific epidemiology phenomena like COVID-19. First of all it's good to know how the simulation works. The simulation works try to mimic a *Social Network Graph*, with each node represents a *Point of Interest (PoI)* where the simulation of the epidemic take place as a system of *Ordinary Differentiable Equation (ODE)*. 

Now we are ready to define our model. We roughly need to know the following set of parameters:

```julia
params = Dict(
	:numNodes => ..., # number of total nodes that make the graph. Default 50
	:edgesCoverage => ..., # try to generalize the how many edges there are
 							# between the nodes. Generally more edges mean 
							# a flux of migration more wide. 
							# If not specified is assumed :high.
							# The possible values are :high, :medium and :low
	:initialNodeInfected => ..., # number of initial node from where the epidemic start. 
								# Default 1
	:param => ..., # a vector that identify the parameters of an epidemic.
					# Typically represents the following variables:
					# R₀, γ = infectivity period, σ = exposed period,
					# ω = immunity period, δ = mortality rate.
					# Default [3.54, 1 / 14, 1 / 5, 1 / 280, 0.01]
	:avgPopulation => ..., # average population from wich the population of
							# each node is created following an exponential distribution.
							# Default 10_000
	:maxTravelingRate => ..., # maximum flux of people from one node to another.
								# This value is used to create a matrix of migration
								# between nodes. Default 0.001
	:control => ..., # boolean value that notify the use of a non-pharmaceutical control.
						# Default false
	:vaccine => ..., # boolean value that notify the simulation of a 
						# random research and than application of a pharmaceutical control.
						# Default false
	:seed => ..., # for reproducibility. Default 1234
)
```

Now that we know the set of parameters useful to run the simulation, we create our first model as follow:
```julia
model = JEPMC.init(; numNodes=8, edgesCoverage=:low, avgPopulation=1000)
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

The property *integrator* of the model is created once the model is instantiated and is used to create an array of **ODEProblem** storing the relative ODE system of each node. Here the initial parameter *:param* is modified adding two extra values η, ξ corresponding to the strictness of the non-pharmaceutical countermeasures applied in a specific node and the vaccine coverage when a vaccine is found. The η value try to summarize very roughly the contermeasures associated to the [OxCGRT project](https://github.com/OxCGRT/covid-policy-tracker).

This additional parameters cannot be modified for now but could be in a future update.

Once the model is being instantiated, it's time to make it run and collect the output. The function that is responsible to that takes as input a bunch of parameters but only a few is really important and useful
```julia
data = JEPMC.collect!(model; # the model we want to simulate
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
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      2  [1.0, 0.0, 0.0, 0.0, 0.0]           0.531899      0.0     ⋯
   2 │     1      2  [1.0, 2.80258e-9, 1.194e-8, 8.64…   0.531899      0.0
   3 │     2      2  [1.0, 1.00011e-8, 2.3532e-8, 3.3…   0.531899      0.0
   4 │     3      2  [1.0, 2.0427e-8, 3.59609e-8, 7.5…   0.531899      0.0
   5 │     4      2  [1.0, 3.35105e-8, 5.00503e-8, 1.…   0.531899      0.0     ⋯
   6 │     5      2  [1.0, 4.9061e-8, 6.64188e-8, 2.1…   0.531899      0.0
   7 │     6      2  [1.0, 6.71311e-8, 8.55802e-8, 3.…   0.531899      0.0
   8 │     7      2  [1.0, 8.79344e-8, 1.08007e-7, 4.…   0.531899      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      2  [0.241657, 0.00285978, 0.0074373…   1.0           0.0     ⋯
 296 │   295      2  [0.243426, 0.00287002, 0.0074224…   1.0           0.0
 297 │   296      2  [0.245188, 0.0028812, 0.00741036…   1.0           0.0
 298 │   297      2  [0.246941, 0.00289331, 0.0074011…   1.0           0.0
 299 │   298      2  [0.248686, 0.00290636, 0.0073946…   1.0           0.0     ⋯
 300 │   299      2  [0.250422, 0.00292037, 0.0073910…   1.0           0.0
 301 │   300      2  [0.252149, 0.00293533, 0.0073901…   1.0           0.0
                                                  2 columns and 286 rows omitted
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      3  [0.999133, 0.0, 0.000867303, 0.0…   0.906939      0.0     ⋯
   2 │     1      3  [0.998921, 0.00019203, 0.0008189…   0.906072      0.0
   3 │     2      3  [0.998725, 0.000342638, 0.000806…   0.905312      0.0
   4 │     3      3  [0.998529, 0.000466489, 0.000821…   0.90462       0.0
   5 │     4      3  [0.998326, 0.000573846, 0.000857…   0.90397       0.0     ⋯
   6 │     5      3  [0.998113, 0.000671957, 0.000910…   0.903341      0.0
   7 │     6      3  [0.997885, 0.00076601, 0.0009775…   0.902721      0.0
   8 │     7      3  [0.997639, 0.000859796, 0.001057…   0.902099      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      3  [0.414329, 0.00501511, 0.0103825…   1.0           0.0     ⋯
 296 │   295      3  [0.415264, 0.00510116, 0.0105428…   1.0           0.0
 297 │   296      3  [0.416177, 0.00518931, 0.0107073…   1.0           0.0
 298 │   297      3  [0.417066, 0.00527961, 0.0108761…   1.0           0.0
 299 │   298      3  [0.417931, 0.00537208, 0.0110493…   1.0           0.0     ⋯
 300 │   299      3  [0.418772, 0.00546678, 0.0112269…   1.0           0.0
 301 │   300      3  [0.419589, 0.00556372, 0.011409,…   1.0           0.0
                                                  2 columns and 286 rows omitted
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      4  [1.0, 0.0, 0.0, 0.0, 0.0]           0.881163      0.0     ⋯
   2 │     1      4  [1.0, 5.0588e-8, 2.15523e-7, 1.5…   0.881163      0.0
   3 │     2      4  [0.999999, 1.80434e-7, 4.24552e-…   0.881163      0.0
   4 │     3      4  [0.999999, 3.68349e-7, 6.48464e-…   0.881163      0.0
   5 │     4      4  [0.999998, 6.03974e-7, 9.02079e-…   0.881162      0.0     ⋯
   6 │     5      4  [0.999998, 8.83806e-7, 1.1965e-6…   0.881162      0.0
   7 │     6      4  [0.999997, 1.20873e-6, 1.54091e-…   0.881161      0.0
   8 │     7      4  [0.999996, 1.58251e-6, 1.94374e-…   0.88116       0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      4  [0.349502, 0.00287548, 0.0067464…   1.0           0.0     ⋯
 296 │   295      4  [0.351147, 0.00289754, 0.0067746…   1.0           0.0
 297 │   296      4  [0.352781, 0.0029205, 0.00680493…   1.0           0.0
 298 │   297      4  [0.354403, 0.00294438, 0.0068373…   1.0           0.0
 299 │   298      4  [0.356014, 0.00296919, 0.0068718…   1.0           0.0     ⋯
 300 │   299      4  [0.357613, 0.00299495, 0.0069085…   1.0           0.0
 301 │   300      4  [0.3592, 0.00302166, 0.00694744,…   1.0           0.0
                                                  2 columns and 286 rows omitted
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      5  [1.0, 0.0, 0.0, 0.0, 0.0]           0.949968      0.0     ⋯
   2 │     1      5  [1.0, 2.96591e-9, 1.26358e-8, 9.…   0.949968      0.0
   3 │     2      5  [1.0, 1.06641e-8, 2.49033e-8, 3.…   0.949968      0.0
   4 │     3      5  [1.0, 2.1751e-8, 3.80826e-8, 8.0…   0.949968      0.0
   5 │     4      5  [1.0, 3.56385e-8, 5.30361e-8, 1.…   0.949968      0.0     ⋯
   6 │     5      5  [1.0, 5.21336e-8, 7.04132e-8, 2.…   0.949968      0.0
   7 │     6      5  [1.0, 7.12976e-8, 9.07549e-8, 3.…   0.949968      0.0
   8 │     7      5  [1.0, 9.33593e-8, 1.1456e-7, 4.6…   0.949968      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      5  [0.301446, 0.00229387, 0.0060140…   1.0           0.0     ⋯
 296 │   295      5  [0.303435, 0.00229406, 0.0059847…   1.0           0.0
 297 │   296      5  [0.305417, 0.00229498, 0.0059578…   1.0           0.0
 298 │   297      5  [0.30739, 0.00229664, 0.00593328…   1.0           0.0
 299 │   298      5  [0.309355, 0.00229904, 0.0059110…   1.0           0.0     ⋯
 300 │   299      5  [0.311313, 0.00230216, 0.0058910…   1.0           0.0
 301 │   300      5  [0.313262, 0.00230602, 0.0058733…   1.0           0.0
                                                  2 columns and 286 rows omitted
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      6  [1.0, 0.0, 0.0, 0.0, 0.0]           0.819254      0.0     ⋯
   2 │     1      6  [1.0, 2.42664e-9, 1.03383e-8, 7.…   0.819254      0.0
   3 │     2      6  [1.0, 8.65962e-9, 2.03756e-8, 2.…   0.819254      0.0
   4 │     3      6  [1.0, 1.76873e-8, 3.11377e-8, 6.…   0.819254      0.0
   5 │     4      6  [1.0, 2.90163e-8, 4.33379e-8, 1.…   0.819254      0.0     ⋯
   6 │     5      6  [1.0, 4.24818e-8, 5.75119e-8, 1.…   0.819254      0.0
   7 │     6      6  [1.0, 5.81294e-8, 7.41046e-8, 2.…   0.819254      0.0
   8 │     7      6  [1.0, 7.6144e-8, 9.3525e-8, 3.79…   0.819254      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      6  [0.297167, 0.00230432, 0.0061071…   1.0           0.0     ⋯
 296 │   295      6  [0.299171, 0.00230291, 0.0060724…   1.0           0.0
 297 │   296      6  [0.301168, 0.00230226, 0.0060402…   1.0           0.0
 298 │   297      6  [0.303157, 0.00230235, 0.0060105…   1.0           0.0
 299 │   298      6  [0.305138, 0.00230317, 0.0059832…   1.0           0.0     ⋯
 300 │   299      6  [0.307111, 0.00230474, 0.0059582…   1.0           0.0
 301 │   300      6  [0.309077, 0.00230704, 0.0059356…   1.0           0.0
                                                  2 columns and 286 rows omitted
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      7  [1.0, 0.0, 0.0, 0.0, 0.0]           0.471452      0.0     ⋯
   2 │     1      7  [0.999997, 2.56142e-6, -4.22375e…   0.471452      0.0
   3 │     2      7  [0.999997, 2.06191e-6, 5.74247e-…   0.471454      0.0
   4 │     3      7  [0.999997, 1.7496e-6, 4.23052e-7…   0.471454      0.0
   5 │     4      7  [0.999997, 1.56916e-6, 7.13876e-…   0.471455      0.0     ⋯
   6 │     5      7  [0.999997, 1.48287e-6, 9.5733e-7…   0.471455      0.0
   7 │     6      7  [0.999996, 1.46515e-6, 1.17269e-…   0.471455      0.0
   8 │     7      7  [0.999996, 1.49884e-6, 1.37365e-…   0.471455      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      7  [0.315073, 0.00230885, 0.0058552…   1.0           0.0     ⋯
 296 │   295      7  [0.317006, 0.00231408, 0.0058416…   1.0           0.0
 297 │   296      7  [0.318929, 0.00232004, 0.0058302…   1.0           0.0
 298 │   297      7  [0.320844, 0.00232672, 0.0058209…   1.0           0.0
 299 │   298      7  [0.32275, 0.00233414, 0.00581367…   1.0           0.0     ⋯
 300 │   299      7  [0.324647, 0.00234229, 0.0058084…   1.0           0.0
 301 │   300      7  [0.326534, 0.00235117, 0.0058053…   1.0           0.0
                                                  2 columns and 286 rows omitted
 301×7 DataFrame
 Row │ step   id     status                             happiness  η        υ  ⋯
     │ Int64  Int64  Array…                             Float64    Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │     0      8  [1.0, 0.0, 0.0, 0.0, 0.0]           0.669388      0.0     ⋯
   2 │     1      8  [1.0, 0.0, 0.0, 0.0, 0.0]           0.669388      0.0
   3 │     2      8  [1.0, 2.23788e-10, 7.46254e-12, …   0.669388      0.0
   4 │     3      8  [1.0, 3.81024e-10, 9.54113e-11, …   0.669388      0.0
   5 │     4      8  [1.0, 5.14968e-10, 2.39797e-10, …   0.669388      0.0     ⋯
   6 │     5      8  [1.0, 6.52498e-10, 4.28588e-10, …   0.669388      0.0
   7 │     6      8  [1.0, 8.10752e-10, 6.5732e-10, 5…   0.669388      0.0
   8 │     7      8  [1.0, 1.0013e-9, 9.26349e-10, 7.…   0.669388      0.0
  ⋮  │   ⋮      ⋮                    ⋮                      ⋮         ⋮        ⋱
 295 │   294      8  [0.227504, 0.00287246, 0.0093615…   1.0           0.0     ⋯
 296 │   295      8  [0.229682, 0.00283747, 0.0091781…   1.0           0.0
 297 │   296      8  [0.231858, 0.00280396, 0.0090024…   1.0           0.0
 298 │   297      8  [0.234032, 0.00277188, 0.0088342…   1.0           0.0
 299 │   298      8  [0.236204, 0.00274118, 0.0086731…   1.0           0.0     ⋯
 300 │   299      8  [0.238373, 0.00271183, 0.0085189…   1.0           0.0
 301 │   300      8  [0.240539, 0.00268378, 0.0083711…   1.0           0.0
                                                  2 columns and 286 rows omitted
```
The last thing we want to do is plot our data
```julia
plt = JEPMC.plot_model(data)
```

## Now let's try something different and use a controller