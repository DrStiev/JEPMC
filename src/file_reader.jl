module file_reader
    using CSV
    using DataFrames, DelimitedFiles
    using DrWatson: @dict

    function extract_param_from_csv(input)
        df = DataFrame(CSV.File(input))
        params = Dict(pairs(eachcol(df)))
        return params
    end
end
