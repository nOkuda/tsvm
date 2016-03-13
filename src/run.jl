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
        "--frac"
            help = "fraction of positive in example"
            arg_type = Float64
            default = 0.5
        "--c"
            help = "penalty weighting for training examples"
            arg_type = Float64
            default = 1.0
        "--cstar"
            help = "penalty weighting for test examples"
            arg_type = Float64
            default = 1.0
    end
    return parse_args(s)
end

function eval(predictions, labels)
    correct = 0
    for i in 1:length(predictions)
        if predictions[i] == labels[i]
            correct += 1
        end
    end
    return correct / length(predictions)
end

function cross_val(data, k, frac, c, cstar)
    if k < 4
        k = 4
    end
    randorder = shuffle(collect(1:length(data.features)))
    testsize = convert(Int, length(data.features) / k)
    result = 0.0
    predictions = train_tsvm(
        randorder[testsize+1:end],
        randorder[1:testsize],
        data,
        frac,
        c,
        cstar,
        false)
    result += eval(predictions, data.labels[randorder[1:testsize]]) / k
    for i in 2:k-1
        trainids = randorder[1:testsize*(i-1)]
        append!(trainids, randorder[(testsize*(i)+1):end])
        testids = randorder[(testsize*(i-1)+1):(testsize*(i))]
        predictions = train_tsvm(
            trainids,
            testids,
            data,
            frac,
            c,
            cstar,
            false)
        result += eval(predictions, data.labels[testids]) / k
    end
    predictions = train_tsvm(
        randorder[1:(testsize*(k-1)+1)],
        randorder[(testsize*(k-1)+1):end],
        data,
        frac,
        c,
        cstar,
        false)
    result += eval(
        predictions, data.labels[randorder[(testsize*(k-1)+1):end]]) / k
    return result
end

function main()
    parsed_args = parse_commandline()
    data = get_data(parsed_args["datafile"])
    cross_val(
        data, 10, parsed_args["frac"], parsed_args["c"], parsed_args["cstar"])
end

main()

