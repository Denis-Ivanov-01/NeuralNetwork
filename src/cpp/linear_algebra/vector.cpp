#include "lin_alg.h"
#include <stdexcept>
#include <format>
#include <iostream>

namespace lin_alg{

Vector::Vector(size_t size) : size(size) {
    if (size < 1) {
        throw std::invalid_argument(std::format("Vector size must be >= 1. Given: {}", size));
    }
    data = std::make_unique<std::vector<double>>(size, 0.0);
}

// Deep Copy Constructor
Vector::Vector(const Vector& other) : size(other.size) {
    data = std::make_unique<std::vector<double>>(*other.data);
}


// Deep Copy Assignment
Vector& Vector::operator=(const Vector& other) {
    if (this == &other) return *this;  // Self-assignment check

    size = other.size;
    data = std::make_unique<std::vector<double>>(*other.data);

    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (this->size != other.size) {
        throw std::invalid_argument("Vector sizes must be equal!");
    }

    for (size_t i = 0; i < size; i++) {
        (i) -= other(i);
    }

    return *this; 
}

Vector& Vector::operator+=(const Vector& other){
    if (this->size != other.size) throw std::invalid_argument("Vector sizes must be equal!");

    for (size_t i = 0; i < size; i++){
        (i) += other(i);
    }

    return *this;
}

Vector Vector::operator*(const Matrix& other) const{
    if (size != other.get_rows_count()){
        throw std::runtime_error("Vector size must be equal to Matrix rows!");
    }

    Vector result(other.get_cols_count());

    for (size_t j = 0; j < other.get_cols_count(); ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            sum += (i) * other(i, j);
        }
        result(j) = sum;
    }

    return result;
}

Vector::Vector(const std::vector<double>& other) : size(other.size()) {
    data = std::make_unique<std::vector<double>>(other);
}

size_t Vector::get_size() const { return size; }

void Vector::validate_index(size_t i) const {
    if (i >= size) {
        throw std::out_of_range("Index out of range");
    }
}

const double Vector::operator()(size_t i) const {
    validate_index(i);
    return (*data)[i];
}

double& Vector::operator()(size_t i) {
    validate_index(i);
    return (*data)[i];
}

// Vector Scalar    ication
Vector Vector::operator*(double scalar) const{
    Vector result(size);
    for (size_t i = 0; i < size; ++i) {
        result(i) = (i) * scalar;
    }
    return result;
}

Vector Vector::operator+(const Vector& other) const{
    if (size != other.size){
        throw std::runtime_error("Vector sizes must be equal");
    }

    Vector result(size);
    for (size_t i = 0; i < size; i++){
        result(i) = (i) + other(i);
    }

    return result;
}

// Vector Transposition (returns a row matrix)
Matrix Vector::transpose() const {
    Matrix result(1, size);
    for (size_t i = 0; i < size; i++) {
        result(0, i) = (i);
    }
    return result;
}

Vector Vector::from_std_vector(const std::vector<double>& v){
    Vector result(v.size());

    for (size_t i = 0; i < result.get_size(); i++){
        result(i) = v[i];
    }

    return result;
}

Vector Vector::from_matrix_row(const Matrix& input, size_t row){
    Vector result(input.get_cols_count());

    for (size_t c = 0; c < result.get_size(); c++){
        result(c) = input(row, c);
    }

    return result;
}

void Vector::print() const{
    for (size_t i = 0; i < size; i++){
        std::cout << (i) << " ";
    }
    std::cout << std::endl;
}

}