module file_reader
    using CSV
    using DataFrames, DelimitedFiles

    input = "csv_files/example.csv"
    df = DataFrame(CSV.File(input))

    # file che si occupa di leggere da un qualche formato,
    # estrapolare i dati e caricarli sotto forma di DataFrame
end
