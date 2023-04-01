module file_reader
    using CSV
    using DataFrames, DelimitedFiles
    using DrWatson: @dict

    # TODO: leggo file csv, poi trasformo in dataframe e infine estrapolo parametri in @dict
    input = "csv_files/example.csv"
    df = DataFrame(CSV.File(input))
end
