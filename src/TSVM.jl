module TSVM

using JuMP
using Ipopt

export TSVMData, train_tsvm, train_svm

type TSVMData
    features
    labels
end

function constrain_vectors!(m, y, w, x, b, xi)
    for i in 1:length(x)
        @addConstraint(m, (y[i] * (dot(w, x[i]) + b)) >= (1 - xi[i]))
        @addConstraint(m, xi[i] >= 0)
    end
end

function initialize_vars!(v, val)
    for i in 1:length(v)
        setValue(v[i], val)
    end
end

function solve_svm_qp(
        training_features,
        training_labels,
        test_features,
        test_predictions,
        C,
        C_star_minus,
        C_star_plus)
    # OP 3 in Joachims, 1999
    m = Model(solver=IpoptSolver(
        print_level=0))
    @defVar(m, weights[1:length(training_features[1])])
    initialize_vars!(weights, 1.0)
    @defVar(m, bias)
    setValue(bias, 1.0)
    @defVar(m, m_training_margin[1:length(training_labels)])
    initialize_vars!(m_training_margin, 1.0)
    constrain_vectors!(
        m, training_labels, weights, training_features, bias, m_training_margin)
    if length(test_predictions) > 0
        @defVar(m, m_test_margin[1:length(test_predictions)])
        initialize_vars!(m_test_margin, 1.0)
        constrain_vectors!(
            m, test_predictions, weights, test_features, bias, m_test_margin)
        @setObjective(
            m, Min, 0.5*dot(weights, weights) + C*sum(m_training_margin) +
            C_star_minus*sum(m_test_margin .* (test_predictions .== -1)) +
            C_star_plus*sum(m_test_margin .* (test_predictions .== 1)))
        status = solve(m)
        if status != :Optimal
            println("NOT OPTIMAL!")
        end
        return [getValue(weights[i]) for i in 1:length(weights)],
            getValue(bias),
            [getValue(m_training_margin[i]) for i in 1:length(m_training_margin)],
            [getValue(m_test_margin[i]) for i in 1:length(m_test_margin)]
    end
    @setObjective(m, Min, 0.5*dot(weights, weights) + C*sum(m_training_margin))
    status = solve(m)
    if status != :Optimal
        println("NOT OPTIMAL!")
    end
    return [getValue(weights[i]) for i in 1:length(weights)],
        getValue(bias),
        [getValue(m_training_margin[i]) for i in 1:length(m_training_margin)],
        []
end

function classify_examples(features::Vector{Vector{AbstractFloat}},
        weights::Vector{Float64}, bias::Float64,
        num_plus::Int64)
    raw_results = [dot(features[i], weights) for i in 1:length(features)] + bias
    pq = Collections.PriorityQueue(1:length(raw_results), raw_results, Base.Order.Reverse)
    ordered_indices = [Collections.dequeue!(pq) for a in 1:length(raw_results)]
    best_indices = ordered_indices[1:num_plus]
    result = -1*ones(Int64, length(features))
    result[best_indices] = 1
    return result
end

function find_problems(
        predictions::Vector{Int64}, xi_star::Vector{Float64}, swapped_dict)
    index1 = -1
    index2 = -1
    found_problem = false
    for i in 1:(length(predictions)-1)
        for j in (i+1):length(predictions)
            if ((predictions[i] * predictions[j] < 0) && (xi_star[i] > 0) &&
                    (xi_star[j] > 0) && (xi_star[i] + xi_star[j] > 2)) &&
                    (get(swapped_dict, (i,j), 0) < 10)
                swapped_dict[(i,j)] = get(swapped_dict, (i,j), 0) + 1
                index1 = i
                index2 = j
                found_problem = true
            end
        end
        if found_problem
            break
        end
    end
    return (index1, index2)
end

function compute_fraction(data, trainids)
    return sum(ones(length(trainids)) .* (data.labels[trainids] .== 1)) /
        length(trainids)
end

function train_svm(
        training_ids::Vector{Int64},
        test_ids::Vector{Int64},
        data::TSVMData,
        c::Float64,
        c_star::Float64,
        debug::Bool)
    plus_percentage = compute_fraction(data, training_ids)
    num_plus = round(Int, plus_percentage*length(test_ids))
    training_features = data.features[training_ids]
    training_labels = data.labels[training_ids]
    (w, b, xi, _) = solve_svm_qp(
        training_features,
        training_labels,
        [],
        [],
        c,
        0,
        0)
    test_features = data.features[test_ids]
    return classify_examples(test_features, w, b, num_plus)
end

function objective_f(w, C, xi, C_star_minus, C_star_plus, xi_star)
    return 0.5 * dot(w, w) + C * sum(xi) +
        C_star_minus * sum(xi_star .* (xi_star .== -1)) +
        C_star_plus * sum(xi_star .* (xi_star .== 1))
end

function train_tsvm(
        training_ids::Vector{Int64},
        test_ids::Vector{Int64},
        data::TSVMData,
        c::Float64,
        c_star::Float64,
        debug::Bool)
    plus_percentage = compute_fraction(data, training_ids)
    num_plus = round(Int, plus_percentage*length(test_ids))
    # based on Figure 4 of Joachims' 1999 transductive SVM paper
    training_features = data.features[training_ids]
    training_labels = data.labels[training_ids]
    (w, b, xi, _) = solve_svm_qp(
        training_features,
        training_labels,
        [],
        [],
        c,
        0,
        0)
    test_features = data.features[test_ids]
    predictions = classify_examples(test_features, w, b, num_plus)
    c_star_minus = 10^-5.0
    c_star_plus = 10^-5.0 * num_plus / (length(test_ids) - num_plus)
    if debug
        message = "Initial"
        println(message)
        print_report(
            message, predictions, w, b, xi, [])
        print_cstars(c_star_plus, c_star_minus)
    end
    count = 1
    while (c_star_minus < c_star) || (c_star_plus < c_star)
        (w, b, xi, xi_star) = solve_svm_qp(
            training_features,
            training_labels,
            test_features,
            predictions,
            c,
            c_star_minus,
            c_star_plus)
        if debug
            message = "Iteration: $(count)"
            println(message)
            print_report(
                message, predictions, w, b, xi, xi_star)
            count += 1
        end
        continue_refinement = true
        swapped_dict = Dict()
        swapped_iter = 0
        while continue_refinement && (swapped_iter < 50)
            swapped_iter += 1
            (index1, index2) = find_problems(predictions, xi_star, swapped_dict)
            if (index1 != -1) && (index2 != -1)
                if debug
                    println("#### Sum: $(xi_star[index1] + xi_star[index2])")
                    print_problems(index1, index2, xi_star)
                end
                predictions[index1] = -1*predictions[index1]
                predictions[index2] = -1*predictions[index2]
                (w, b, xi, xi_star) = solve_svm_qp(
                    training_features,
                    training_labels,
                    test_features,
                    predictions,
                    c,
                    c_star_minus,
                    c_star_plus)
                if debug
                    message = "Iteration: $(count)\nSwapped $(index1) and $(index2)"
                    println(message)
                    print_report(
                        message, predictions, w, b, xi, xi_star)
                    count += 1
                end
            else
                continue_refinement = false
            end
        end
        c_star_minus = min(c_star_minus*2, c_star)
        c_star_plus = min(c_star_plus*2, c_star)
        if debug
            print_cstars(c_star_plus, c_star_minus)
        end
    end
    return predictions
end

function print_report(
        message, predictions, w, b, xi, xi_star)
    println("########################################")
    println("$(message)")
    println("predictions:\t$(predictions)")
    println("weights:\t$(w)")
    println("bias:\t$(b)")
    println("training penalties:\t$(xi)")
    println("testing penalties:\t$(xi_star)")
    println("#--------------------------------------#")
end

function print_cstars(c_star_plus, c_star_minus)
    println("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    println("\tC_star_plus:\t$(c_star_plus)")
    println("\tC_star_minus:\t$(c_star_minus)")
    println("C**************************************C")
end

function print_problems(index1, index2, xi_star)
    println("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    println("\tindex1:\t$(index1)\t$(xi_star[index1])")
    println("\tindex2:\t$(index2)\t$(xi_star[index2])")
    println("I**************************************I")
end

end
