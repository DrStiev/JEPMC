using Test, Dates, DataFrames, Plots #, Distributed
# addprocs(Int(Sys.CPU_THREADS / 4))
# @everywhere include("../src/JEPMC.jl")
include("../src/JEPMC.jl")

function complex_filter(x, y, z, w, k, j, h)
    x == val[1] &&
        y == val[2] &&
        z == val[3] &&
        w == val[4] &&
        k == val[5] &&
        j == val[6] &&
        h == val[7]
end

function complex_filter(x, y, z, w, k, j)
    x == val[1] && y == val[2] && z == val[3] && w == val[4] && k == val[5] && j == val[6]
end

function complex_filter(x, y, z, w, k)
    x == val[1] && y == val[2] && z == val[3] && w == val[4] && k == val[5]
end

function complex_filter(x, y, z, w)
    x == val[1] && y == val[2] && z == val[3] && w == val[4]
end

function complex_filter(x, y, z)
    x == val[1] && y == val[2] && z == val[3]
end

function complex_filter(x, y)
    x == val[1] && y == val[2]
end

function complex_filter(x)
    x == val[1]
end

val = nothing

function test_paramscan_abm(path::String, properties)
    data = JEPMC.collect_paramscan!(properties)
    plts = []
    global val = nothing
    for i in 1:size(data[2], 1)
        namesz = names(data[2][i, :])
        global val = data[2][i, :]
        # hardcoded but functional
        df = filter(namesz => complex_filter, data[1])
        select!(df, Not(namesz))
        dd = [filter(:id => ==(i), df) for i in unique(df[!, :id])]
        r = []
        for i in 1:size(names(val), 1)
            push!(r, names(val)[i] * ": " * string(val[i]))
        end
        j = join(r, ", ")
        push!(plts, JEPMC.plot_model(dd; title = j))
    end
    JEPMC.save_dataframe(data[1], path, "SocialNetworkABM")
    i = 1
    for plt in plts
        JEPMC.save_plot(plt, path, "SocialNetworkABM_$i", "pdf")
        i += 1
    end
    return true
end
