#include "neural_network.h"


namespace neural_network{
    
std::vector<TrainingBatch> NeuralNetwork::create_batches(const std::vector<TrainingSample>& training_data) {
    std::vector<TrainingBatch> batches;

    size_t offset = 0;

    while (offset < training_data.size()) {
        TrainingBatch batch = create_single_batch(training_data, offset);
        batches.push_back(batch);
        offset += batch.inputs.get_rows_count();
    }
    return batches;
}

TrainingBatch NeuralNetwork::create_single_batch(const std::vector<TrainingSample>& training_data, size_t offset) {
    size_t remaining = training_data.size() - offset;
    size_t curr_batch_size = remaining >= batch_size ? batch_size : remaining;

    TrainingBatch batch(
        lin_alg::Matrix(curr_batch_size, layers.front().get_weights().get_rows_count()),
        lin_alg::Matrix(curr_batch_size, layers.back().get_biases().get_size())
    );
    // batch.inputs = lin_alg::Matrix(curr_batch_size, layers.front().get_weights().get_rows_count());
    // batch.expected_outputs = lin_alg::Matrix(curr_batch_size, layers.back().get_biases().get_size());

    for (size_t i = 0; i < curr_batch_size; ++i) {
        const TrainingSample& ts = training_data[offset + i];

        for (size_t c = 0; c < batch.inputs.get_cols_count(); ++c) {
            batch.inputs(i, c) = ts.input_data[c];
        }

        for (size_t c = 0; c < batch.expected_outputs.get_cols_count(); ++c) {
            batch.expected_outputs(i, c) = ts.expected_output[c];
        }
    }

    return batch;
}
}

