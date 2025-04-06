#pragma once
#include <cmath>


namespace neural_network{

    class ActivationFunc {
        public:
            virtual double apply(double input) = 0;  // Forward pass
            virtual double applyDerivative(double input) = 0;  // Derivative for backpropagation
            virtual ~ActivationFunc() = default;  
        };
        
        class ReLU : public ActivationFunc {
        public:
            double apply(double input) override {
                return input > 0 ? input : 0;  // ReLU: max(0, x)
            }
        
            double applyDerivative(double input) override {
                return input > 0 ? 1 : 0;  // ReLU derivative: 1 if x > 0, else 0
            }
        };
        
        class Sigmoid : public ActivationFunc {
        public:
            double apply(double input) override {
                return 1 / (1 + exp(-input));  // Sigmoid: 1 / (1 + e^(-x))
            }
        
            double applyDerivative(double input) override {
                double sigmoid_value = apply(input);
                return sigmoid_value * (1 - sigmoid_value);  // Sigmoid derivative: σ(x) * (1 - σ(x))
            }
        };
}