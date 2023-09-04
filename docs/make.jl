push!(LOAD_PATH, "../src/")
using JEPMC
using Documenter

makedocs(sitename = "JEPMC.jl",
    modules = [JEPMC],
    pages = ["Home" => "index.md"])

deploydocs(;
    repo = "https://github.com/DrStiev/JEPMC.git",
    devbranch = "main")
