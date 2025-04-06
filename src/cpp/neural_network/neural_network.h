#pragma once

#include <vector>
#include "../linear_algebra/lin_alg.h"
#include "activation_funcs.h"

namespace neural_network{

    #define PRINTN(x)std::cout << x << std::endl;
    #define PRINT_DEBUG(x)std::cout << x << std::endl;

    /// @brief Represents a single training sample for an iteration:
    /// the inputs for each input neuron and the expected output for each output neuron
    struct TrainingSample{
        std::vector<double> input_data;
        std::vector<double> expected_output;
    };

    struct TrainingBatch{
        lin_alg::Matrix inputs;
        lin_alg::Matrix expected_outputs;
    };

    /// @brief A set of parameters between two neuron layers - the weight between the neurons of the n and n+1 layer 
    /// and the biases of the n+1 layer 

    class NNLayer{

        private:
            size_t size;
        
            lin_alg::Matrix weights;
            lin_alg::Vector biases;
            
            std::shared_ptr<ActivationFunc> activate_function;

        public:
        
            NNLayer(size_t input_size, size_t output_size, std::shared_ptr<ActivationFunc> act_func);
        
            void initialize_params();

            lin_alg::Vector forward(const lin_alg::Vector& input) const;  
            ForwardResult forward(const lin_alg::Matrix& input);

            void update(const lin_alg::Matrix& weight_grad, const lin_alg::Vector& bias_grad);

            lin_alg::Matrix get_weights() const { return weights; }
            lin_alg::Vector get_biases() const { return biases; }

            std::shared_ptr<ActivationFunc> get_activation() const;

            lin_alg::Matrix& expose_weights();
            lin_alg::Vector& expose_biases();

        };
        
        
    class NeuralNetwork{
        private:

            std::vector<NNLayer> layers;

            //parameters of the learning process
            // double learning_rate;
            int batch_size;

            std::vector<lin_alg::Matrix> activations;
            std::vector<lin_alg::Matrix> outputs;

            std::vector<TrainingBatch> NeuralNetwork::create_batches(const std::vector<TrainingSample>& training_data);
            TrainingBatch NeuralNetwork::create_single_batch(const std::vector<TrainingSample>& training_data, size_t offset);

            //forward calculations
            lin_alg::Vector predict(const lin_alg::Vector& input) const;
            lin_alg::Matrix forward(const lin_alg::Matrix& input_batch);

            void backward(const TrainingBatch& batch, double learning_rate);

            double calc_rmse(const std::vector<lin_alg::Vector>& predicted_vals, const std::vector<lin_alg::Vector>& target_vals) const;

            double calc_correlation(const std::vector<lin_alg::Vector>& predicted_vals, const std::vector<lin_alg::Vector>& target_vals) const;

            lin_alg::Vector normalize_input(const lin_alg::Vector& input) const;

            lin_alg::Matrix normalize_inputs(const lin_alg::Matrix& inputs) const;

            lin_alg::Vector denormalize_output(const lin_alg::Vector& output) const;

        public:

        NeuralNetwork(int batch_size);

        void train(std::vector<TrainingSample>& training_data, int epochs, double learning_rate);

        void test(const std::vector<TrainingSample>& test_data) const;
        };        
}
