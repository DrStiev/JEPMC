using Test, Dates

include("../src/SocialNetworkABM.jl")
include("../src/ABMUtils.jl")

function test_sensitivity(path::String)
    x, dp, pltout = sensitivity_analisys(seir!,
        [0.999, 0.0, 0.001, 0.0, 0.0],
        (0.0, 1200.0),
        [3.54, 1 / 14, 1 / 5, 1 / 280, 0.001, 0.0, 0.0])
    JEPMC.save_plot(pltout, path, "sensitivity_analysis")
    return true
end
