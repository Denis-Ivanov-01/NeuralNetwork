#include "neural_network.h"
#include <random>

namespace neural_network{
        // Constructor initializes weights and biases
        NNLayer::NNLayer(size_t input_size, size_t output_size, std::shared_ptr<ActivationFunc> act_func)
        : weights(input_size, output_size), biases(output_size), activate_function(act_func) {
        initialize_params();
    }

    std::shared_ptr<ActivationFunc> NNLayer::get_activation() const{
        return activate_function;
    }

    // Randomly initializes weights and biases
    void NNLayer::initialize_params() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-1.0, 1.0);

        for (size_t i = 0; i < weights.get_rows_count(); ++i)
            for (size_t j = 0; j < weights.get_cols_count(); ++j)
                weights(i, j) = dist(gen);
                // weights(i, j) = 0.1;

        for (size_t i = 0; i < biases.get_size(); ++i)
            biases(i) = dist(gen);
            // biases(i) = 0.1;
    }

    // Forward pass through the layer
    lin_alg::Vector NNLayer::forward(const lin_alg::Vector& input) const{
        lin_alg::Vector z = (input * weights) + biases;
        for (size_t i = 0; i < z.get_size(); ++i)
            z(i) = (*activate_function).apply(z(i));
        return z;
    }

    ForwardResult NNLayer::forward(const lin_alg::Matrix& input){
        lin_alg::Matrix z = input * weights.transpose();
        
        lin_alg::Matrix output(z.get_rows_count(), z.get_cols_count());
        for (size_t r = 0; r < z.get_rows_count(); r++){
            for (size_t c = 0; c < z.get_cols_count(); c++){
                // add the biases to each column of every row - every row is a different sample
                z(r, c) = z(r, c) + biases(c);
                output(r, c) = z(r, c);
            }
        }
        std::function<double(double)> func = [&](double x) {return (*activate_function).apply(x);};
        
        output.apply_to_elements(func);

        ForwardResult result{z, output};
        return result;
    }

    lin_alg::Matrix& NNLayer::expose_weights(){
        return this->weights;
    }

    lin_alg::Vector& NNLayer::expose_biases(){
        return this->biases;
    }

    // Update weights and biases using gradients
    void NNLayer::update(const lin_alg::Matrix& weight_grad, const lin_alg::Vector& bias_grad) {
        weights += weight_grad;
        biases += bias_grad;
    }


}
