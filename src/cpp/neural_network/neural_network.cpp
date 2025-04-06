#include "neural_network.h"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <format>

#define assertm(exp, msg) assert((void(msg), exp))

namespace neural_network {

    NeuralNetwork::NeuralNetwork(int batch_size) : batch_size(batch_size) {
        std::shared_ptr<ActivationFunc> sigmoid = std::make_shared<Sigmoid>();
            std::shared_ptr<ActivationFunc> relu = std::make_shared<ReLU>();

            //register input layer
            layers.push_back(NNLayer(4, 10, sigmoid));
            
            //hidden layers
            layers.push_back(NNLayer(10, 6, sigmoid));
        
            //output layer
            layers.push_back(NNLayer(6, 1, sigmoid));
    }

    lin_alg::Vector NeuralNetwork::normalize_input(const lin_alg::Vector& input) const {
        lin_alg::Vector normalized(input.get_size());
    
        // Normalize the first input (0 to 30)
        normalized(0) = input(0) /30;
    
        // Normalize the second input (0 to 10)
        normalized(1) = input(1) / 10;
    
        // Normalize the third input (0 to 30)
        normalized(2) = input(2) / 30;
    
        // Normalize the fourth input (0 to 10)
        normalized(3) = input(3) / 10;
    
        return normalized;
    }
    
    // Function to normalize an entire matrix of inputs (by rows)
    lin_alg::Matrix NeuralNetwork::normalize_inputs(const lin_alg::Matrix& inputs) const {
        lin_alg::Matrix normalized(inputs.get_rows_count(), inputs.get_cols_count());

        for (size_t i = 0; i < inputs.get_rows_count(); ++i) {
            lin_alg::Vector input_row = lin_alg::Vector::from_matrix_row(inputs, i);
            lin_alg::Vector normalized_row = normalize_input(input_row);  // Use the previously defined normalize_input function
            
            for (size_t j = 0; j < normalized_row.get_size(); ++j) {
                normalized(i, j) = normalized_row(j);
            }
        }

        return normalized;
    }

    // Example if output was scaled between 0 and some max value
    lin_alg::Vector NeuralNetwork::denormalize_output(const lin_alg::Vector& output) const {
        lin_alg::Vector denormalized(output.get_size());

        // Assuming the output should be denormalized to a different range (like 0 to 100 for example)
        for (size_t i = 0; i < output.get_size(); ++i) {
            denormalized(i) = output(i) * 1.0;  // Since output is already in range [0, 1], no changes needed.
        }

        return denormalized;
    }

    lin_alg::Matrix NeuralNetwork::forward(const lin_alg::Matrix& input){
        activations.clear();
        outputs.clear();

        outputs.push_back(input);


        for (size_t i = 0; i < layers.size(); i++){
            const NNLayer& layer = layers[i];
            lin_alg::Matrix act = (outputs[i] * layer.get_weights()).elementwise_add(layer.get_biases());

            std::function<double(double)> func = [&](double x) {return layer.get_activation()->apply(x);};
            lin_alg::Matrix out = act.apply_to_elements(func);  
            outputs.push_back(out);

        }

        return outputs.back();
    }

    lin_alg::Vector NeuralNetwork::predict(const lin_alg::Vector& input) const{
        lin_alg::Vector result = input;

        for (size_t i = 0; i < layers.size(); i++){
            result = layers[i].forward(result);
        }
        
        return result;
    }

    double NeuralNetwork::calc_rmse(const std::vector<lin_alg::Vector>& predicted_vals, 
        const std::vector<lin_alg::Vector>& target_vals) const {
        // Ensure both vectors have the same number of elements
        if (predicted_vals.size() != target_vals.size()) {
            throw std::invalid_argument("Predictions and targets must have the same number of elements");
        }

        double square_err_sum = 0.0;
        size_t num_elements = 0;

        // Iterate through predictions
        for (size_t i = 0; i < predicted_vals.size(); ++i) {
            const lin_alg::Vector& predicted = predicted_vals[i];
            const lin_alg::Vector& target = target_vals[i];

            // Ensure each vector has the same size
            if (predicted.get_size() != target.get_size()) {
                throw std::invalid_argument("Prediction and target vectors must have the same size");
            }

            // Compute squared error
            for (size_t j = 0; j < predicted.get_size(); ++j) {
                double error = predicted(j) - target(j);
                square_err_sum += error * error;
            }

            num_elements += predicted.get_size();  
        }

        // Prevent division by zero
        if (num_elements == 0) {
            throw std::runtime_error("RMSE calculation: No elements to process (empty input).");
        }

        return std::sqrt(square_err_sum / num_elements);  // Compute and return RMSE
    }


    double NeuralNetwork::calc_correlation(const std::vector<lin_alg::Vector>& predicted_vals,
         const std::vector<lin_alg::Vector>& target_vals) const {
        // Ensure both vectors have the same number of elements
        if (predicted_vals.size() != target_vals.size()) {
        throw std::invalid_argument("Predictions and targets must have the same number of elements");
        }

        size_t num_elements = 0;
        double sum_predicted = 0.0;
        double sum_target = 0.0;
        double sum_predicted_squared = 0.0;
        double sum_target_squared = 0.0;
        double sum_product = 0.0;

        // First pass: Compute sums
        for (size_t i = 0; i < predicted_vals.size(); ++i) {
        const lin_alg::Vector& predicted = predicted_vals[i];
        const lin_alg::Vector& target = target_vals[i];

        if (predicted.get_size() != target.get_size()) {
        throw std::invalid_argument("Prediction and target vectors must have the same size");
        }

        for (size_t j = 0; j < predicted.get_size(); ++j) {
        double pred = predicted(j);
        double targ = target(j);

        sum_predicted += pred;
        sum_target += targ;
        sum_predicted_squared += pred * pred;
        sum_target_squared += targ * targ;
        sum_product += pred * targ;
        }

        num_elements += predicted.get_size();
        }

        if (num_elements == 0) {
        throw std::runtime_error("Correlation calculation: No elements to process (empty input).");
        }

        // Compute means
        double mean_predicted = sum_predicted / num_elements;
        double mean_target = sum_target / num_elements;

        // Compute covariance and variances
        double covariance = (sum_product / num_elements) - (mean_predicted * mean_target);
        double variance_predicted = (sum_predicted_squared / num_elements) - (mean_predicted * mean_predicted);
        double variance_target = (sum_target_squared / num_elements) - (mean_target * mean_target);

        // Check for zero variance
        if (variance_predicted == 0 || variance_target == 0) {
        return 0.0;  // No correlation if either variance is zero
        }

        // Pearson correlation coefficient
        double correlation = covariance / (std::sqrt(variance_predicted) * std::sqrt(variance_target));

        return correlation;
        }    

    void NeuralNetwork::train(std::vector<TrainingSample>& training_data, int epochs, double learning_rate) {
        // PRINTN("weights before")
        // layers.front().get_weights().print_matrix();
        std::vector<TrainingBatch> batches = create_batches(training_data);
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::random_device rd;  // Get a random seed from the OS
            std::mt19937 g(rd());   // Use Mersenne Twister PRNG

            // Shuffle the data
            std::shuffle(training_data.begin(), training_data.end(), g);
            for (const TrainingBatch& batch : batches) {

                lin_alg::Matrix normalized_inputs(batch.inputs.get_rows_count(), batch.inputs.get_cols_count());
                
                //normalize inputs
                for (size_t i = 0; i < batch.inputs.get_rows_count(); ++i) {
                    lin_alg::Vector input_row = lin_alg::Vector::from_matrix_row(batch.inputs, i);
                    lin_alg::Vector normalized_input = normalize_input(input_row);
                    for (size_t j = 0; j < normalized_input.get_size(); ++j) {
                        normalized_inputs(i, j) = normalized_input(j);
                    }
                }
                lin_alg::Matrix out = forward(batch.inputs);
                backward(batch, learning_rate);
        }
    }
    // PRINTN("weights after")
    // layers.front().get_weights().print_matrix();
}

    void NeuralNetwork::test(const std::vector<TrainingSample>& test_data) const{
        std::vector<lin_alg::Vector> predicted_results;
        
        std::vector<lin_alg::Vector> expected_results;
        for (const TrainingSample& sample : test_data){

            lin_alg::Vector in = normalize_input(lin_alg::Vector::from_std_vector(sample.input_data));
            lin_alg::Vector expected = lin_alg::Vector::from_std_vector(sample.expected_output);
            predicted_results.push_back(denormalize_output(predict(in)));
            expected_results.push_back(expected);
        }

        PRINTN("")
        PRINTN("Model accuracy:")
        
        double rmse = calc_rmse(predicted_results, expected_results);
        PRINTN("RMSE: " << rmse)
        
        double corr = calc_correlation(predicted_results, expected_results);
        PRINTN("Correlation: " << corr)
    }

    void NeuralNetwork::backward(const TrainingBatch& batch, double learning_rate){
        std::vector<lin_alg::Matrix> deltas;
        lin_alg::Matrix init_err = batch.expected_outputs - outputs.back();
        std::function<double(double)> func = [&](double x) {return layers.back().get_activation()->applyDerivative(x);};
        lin_alg::Matrix init_delta = init_err.elementwise_mult(outputs.back().apply_to_elements(func));

        deltas.push_back(init_delta);
        
        for (size_t i = layers.size() - 1; i > 0; i--){

            NNLayer& layer = layers[i];
            lin_alg::Matrix prevDelta = deltas.back();
            lin_alg::Matrix weightsT = layer.get_weights().transpose();

            assertm(prevDelta.get_cols_count() == weightsT.get_rows_count(), "Delta cols and weightT rows are not equal");

            lin_alg::Matrix err = prevDelta * weightsT;

            std::function<double(double)> func = [&](double x) {return layer.get_activation()->applyDerivative(x);};
            lin_alg::Matrix delta = err.elementwise_mult(outputs[i].apply_to_elements(func));

            deltas.push_back(delta);
        }


        assertm(layers.size() == deltas.size(), "Deltas size must be equal to layers size!");
        std::reverse(deltas.begin(), deltas.end());

        for (size_t i = 0; i < layers.size(); i++){
            NNLayer& layer = layers[i];
            lin_alg::Matrix delta = deltas[i];
            lin_alg::Matrix weightDiff = (outputs[i].transpose() * delta) * learning_rate;
            lin_alg::Vector biasDiff = delta.collapse_rows() * learning_rate;
            layer.update(weightDiff, biasDiff);
        }
    }
}
