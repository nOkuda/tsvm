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

function evaluate(predictions, labels)
    correct = 0
    for i in 1:length(predictions)
        if predictions[i] == labels[i]
            correct += 1
        end
    end
    return correct / length(predictions)
end

function cross_val(data, k, c, cstar)
    if k < 4
        k = 4
    end
    results = []
    randorder = shuffle(collect(1:length(data.features)))
    testsize = trunc(Int, length(data.features) / k)
    trainids = randorder[testsize+1:end]
    testids = randorder[1:testsize]
    predictions = train_tsvm(
        trainids,
        testids,
        data,
        c,
        cstar,
        false)
    push!(
        results, evaluate(predictions, data.labels[randorder[1:testsize]]))
    for i in 2:k-1
        trainids = randorder[1:testsize*(i-1)]
        append!(trainids, randorder[(testsize*(i)+1):end])
        testids = randorder[(testsize*(i-1)+1):(testsize*(i))]
        predictions = train_tsvm(
            trainids,
            testids,
            data,
            c,
            cstar,
            false)
        push!(results, evaluate(predictions, data.labels[testids]))
    end
    trainids = randorder[1:(testsize*(k-1)+1)]
    testids = randorder[(testsize*(k-1)+1):end]
    predictions = train_tsvm(
        trainids,
        testids,
        data,
        c,
        cstar,
        false)
    push!(results, evaluate(
        predictions, data.labels[randorder[(testsize*(k-1)+1):end]]))
    return results
end

function small_test(data, k, c, cstar)
    trainsize = convert(Integer, trunc(0.1 * length(data.features)))
    results = []
    for _ in 1:k
        randorder = shuffle(collect(1:length(data.features)))
        predictions = train_tsvm(
            randorder[1:trainsize],
            randorder[trainsize+1:end],
            data,
            c,
            cstar,
            false
        )
        push!(results, evaluate(
            predictions, data.labels[randorder[trainsize+1:end]]))
    end
    return results
end

function main()
    parsed_args = parse_commandline()
    data = get_data(parsed_args["datafile"])
    k = 4
    results = cross_val(data, k, parsed_args["c"], parsed_args["cstar"])
    # results = small_test(data, k, parsed_args["c"], parsed_args["cstar"])
    println(results)
    println(sum(results)/k)
end

main()

