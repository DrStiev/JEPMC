### -*- Mode: Julia -*-

### JEPMC.jl

### MIT License

### Copyright (c) 2023 Matteo Stievano

### Permission is hereby granted, free of charge, to any person obtaining a copy
### of this software and associated documentation files (the "Software"), to deal
### in the Software without restriction, including without limitation the rights
### to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
### copies of the Software, and to permit persons to whom the Software is
### furnished to do so, subject to the following conditions:

### The above copyright notice and this permission notice shall be included in all
### copies or substantial portions of the Software.

### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
### IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
### FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
### AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
### LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
### OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
### SOFTWARE.

module JEPMC

# included files
include("SocialNetworkABM.jl")
include("ABMUtils.jl")
include("Utils.jl")
include("Controller.jl")

# exported functions
export init, collect!, ensemble_collect!, collect_paramscan!, plot_system_graph
export plot_system_graph, plot_model
export save_dataframe,
    save_plot,
    save_parameters,
    read_dataset,
    dataset_from_location,
    download_dataset,
    load_parameters

"""
        Julia Epidemiology Model and Control (JEPMC)

        JEPMC is a library designed for the exploration of epidemic
            models and their corresponding control mechanisms.

        # Overview of the Library

        The primary focus of this library is to combine the capabilities
        and flexibility of an Agent-Based simulation using the
        *Agents.jl* framework with the stability and performance
        offered by classical mathematical models defined through
        systems of Ordinary Differential Equations (ODEs).
        Additionally, JEPMC harnesses the power of Neural Networks
        to extrapolate and learn the underlying mechanisms essential
        for epidemic control.

        # Epidemic Model Description

        The model implemented in JEPMC is relatively straightforward
        and emulates a social graph structure. This approach is
        particularly valuable for simulation purposes.
        The central concept involves modeling a network of Points
        of Interest (PoI), where each PoI simulates a node within
        the graph. These nodes are interconnected to varying degrees.
        Each node has its own system of defining ODEs,
        and during each simulation step, one integration step
        corresponds to the ODE system's evolution.

        This approach may initially seem overly deterministic and
        lacking the stochastic behavior typically associated with
        simulations that produce emergent behavior. However,
        the way the model is formulated introduces a touch of
        stochasticity that proves beneficial, as discussed further below.

        Finally, the model incorporates a NeuralODE, responsible for
        identifying and learning the appropriate countermeasures to
        apply in dynamic situations that change over time.
        A dedicated section will delve deeper into this aspect.
"""

end

### end of file -- JEPMC.jl
