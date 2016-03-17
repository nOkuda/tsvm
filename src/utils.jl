module utils
using TSVM


export get_data

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

end
