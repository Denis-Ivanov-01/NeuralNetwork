
#include <iostream>
#include "linear_algebra/lin_alg.h"
#include "neural_network/neural_network.h"
#include "file/reader.h"


int main(){
    file_handling::FileReader reader("C:\\Users\\denis\\Desktop\\Геодезия\\Невронни мрежи\\test_data\\0.1-training.txt");

    std::vector<neural_network::TrainingSample> samples = reader.readTrainingData();
    
    // Testing with LITERALLY the same dataset and still cannot get it to work :/
    file_handling::FileReader test_data_reader("C:\\Users\\denis\\Desktop\\Геодезия\\Невронни мрежи\\test_data\\12.1.txt");
    std::vector<neural_network::TrainingSample> test_samples = test_data_reader.readTrainingData();

    int batch_size(20);
    neural_network::NeuralNetwork network(batch_size);

    try{
        network.test(samples);
        network.train(samples, 100, 0.25);
        network.test(samples);
        network.train(samples, 500, 0.25);
        network.test(samples);
        network.train(samples, 500, 0.15);
        network.test(samples);
        network.train(samples, 500, 0.15);
        network.test(samples);
        network.train(samples, 500, 0.05);
        network.test(samples);
        
    }
    catch (const std::exception& e){
        PRINT("Exception...")
        PRINT(e.what())
    }
    
    do 
    {
    std::cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
}
