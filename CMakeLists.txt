cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory for all build types
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "C:/Users/denis/source/repos/NuralNetwork1/build")
foreach(OUTPUTCONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "C:/Users/denis/source/repos/NuralNetwork1/build")
endforeach()

# Add executable
add_executable(NeuralNetwork
    src/cpp/main.cpp
    src/cpp/linear_algebra/matrix.cpp
    src/cpp/linear_algebra/vector.cpp
    src/cpp/neural_network/neural_network.cpp
    src/cpp/neural_network/layer.cpp
    src/cpp/neural_network/batches.cpp
)

# Include directories
target_include_directories(NeuralNetwork PRIVATE src/cpp)
