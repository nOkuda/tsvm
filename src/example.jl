using ArgParse
include("TSVM.jl")
using TSVM
include("utils.jl")
using utils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "datafile"
            help = "file where data is kept"
            required = true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    data = get_data(parsed_args["datafile"])
    predictions = train_tsvm([1, 6], collect(2:5), data, 1.0, 1.0, true)
    println("final predictions: $(predictions)")
end

main()

