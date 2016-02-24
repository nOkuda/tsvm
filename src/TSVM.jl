module TSVM

using Collections
using JuMP

export TSVMData, train_tsvm

type TSVMData
    features
    labels
end

function constrain_vectors(m, y, w, x, b, xi)
    for i in 1:length(x)
        @addConstraint(m, y[i] * (dot(w, x[i]) + b) >= 1 - xi[i])
    end
end

function solve_svm_qp(training_features, training_labels, test_features,
        test_predictions, C, C_star_minus, C_star_plus)
    # OP 3 in Joachims, 1999
    m = Model()
    @defVar(m, weights[1:length(training_features[1])])
    weights = randn(length(weights))
    @defVar(m, bias)
    bias = 0
    @defVar(m, m_training_features[1:length(training_features),1:length(training_features[1])])
    m_training_features = copy(training_features)
    @defVar(m, m_training_labels[1:length(training_labels)])
    m_training_labels = copy(training_labels)
    @defVar(m, m_training_margin[1:length(training_labels)])
    m_training_margin = rand(length(training_labels))
    constrain_vectors(m, m_training_labels, weights, m_training_features, bias,
        m_training_margin)
    if length(test_predictions) > 0
    end
end

function classify_examples(features::Matrix{AbstractFloat},
        weights::Vector{AbstractFloat}, bias::AbstractFloat,
        num_plus)
    raw_results = *(features, weights) + bias
    pq = PriorityQueue(1:num_plus, raw_results, Base.Order.Reverse)
    best_indices = [dequeue!(pq) for a in 1:num_plus]
    result = -1*ones(length(features))
    result[best_indices] = 1
    return result
end

function find_problems(predictions::Vector{Int}, xi_star::Vector{AbstractFloat})
    index1 = -1
    index2 = -1
    found_problem = false
    for i in 1:length(predictions)-1
        for j in 2:length(predictions)
            if (predictions[i] * predictions[j] < 0) && (xi_star[j] > 0) &&
                    (xi_star[i] + xi_star[j] > 2)
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

function train_tsvm(training_ids::Vector{Integer}, test_ids::Vector{Integer},
        data::TSVMData, plus_percentage, c, c_star)
    num_plus = round(Int, plus_percentage*length(test_ids))
    # based on Figure 4 of Joachims' 1999 transductive SVM paper
    training_features = data.features[training_ids]
    training_labels = data.labels[training_ids]
    (w, b, xi, _) = solve_svm_qp(training_features,
            training_labels, [], [], c, c_star)
    test_features = data.features[test_ids]
    predictions = classify_examples(test_features, w, b, num_plus)
    c_star_minus = 10^-5
    c_star_plus = 10^-5 * num_plus / (length(test_ids) - num_plus)
    while (c_star_minus < c_star) || (c_star_plus < c_star)
        (w, b, xi, xi_star) = solve_svm_qp(training_features,
                training_labels, test_features,
                predictions, c, c_star_minus, c_star_plus)
        continue_refinement = true
        while continue_refinement
            (index1, index2) = find_problems(predictions, xi_star)
            if (index1 != -1) && (index2 != -1)
                predictions[index1] = -1*predictions[index1]
                predictions[index2] = -1*predictions[index2]
                solve_svm_qp(training_features, training_labels, test_features,
                        predictions, c, c_star_minus, c_star_plus)
            else
                continue_refinement = false
            end
        end
        c_star_minus = min(c_star_minus*2, c_star)
        c_star_plus = min(c_star_plus*2, c_star)
    end
    return predictions
end

end
