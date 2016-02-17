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
    return data
end

function main()
    parsed_args = parse_commandline()
    println("$(parsed_args["datafile"])")
    data = get_data(parsed_args["datafile"])
end

main()

