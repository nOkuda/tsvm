using ArgParse
include("TSVM.jl")
using TSVM

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "datafile"
            help = "file where data is kept"
            required = true
    end
    return parse_args(s)
end

function get_data(filename)
    data = Array{AbstractFloat}[]
    open(filename) do fh
        for line in eachline(fh)
            push!(data, [parse(Float64, a) for a in split(strip(line),
                    ','; keep=false)])
        end
    end
    data_length = length(data[1])
    return TSVMData(
        [a[1:data_length-1] for a in data],
        [a[data_length] for a in data])
end

function main()
    parsed_args = parse_commandline()
    println("$(parsed_args["datafile"])")
    data = get_data(parsed_args["datafile"])
    predictions = train_tsvm([1, 6], collect(2:5), data, 0.5, 1.0, 1.0)
    println(predictions)
end

main()

