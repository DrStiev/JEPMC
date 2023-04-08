module collect_data
    using Agents

    function collect(model, astep, mstep; n = 1000)
        susceptible(x) = count(i == :S for i in x)
        infected(x) = count(i == :I for i in x)
        recovered(x) = count(i == :R for i in x)

        to_collect = [(:status, f) for f in (susceptible, infected, recovered, length)]
        data, _ = run!(model, astep, mstep, n; adata = to_collect)
        return data
    end
end