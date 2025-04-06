#include "lin_alg.h"
#include <iostream>
#include <stdexcept>
#include <format>
#include <functional>

namespace lin_alg {


Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    if (rows < 1 || cols < 1) {
        std::string message = std::format("Invalid rows or columns! Rows: {}, Columns: {}", rows, cols);
        throw std::invalid_argument(message);
    }
    data = std::make_unique<std::vector<double>>(rows * cols, 0);
}

// Deep Copy Constructor
Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    data = std::make_unique<std::vector<double>>(*other.data);
}

// Deep Copy Assignment
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;  // Self-assignment check

    rows = other.rows;
    cols = other.cols;
    data = std::make_unique<std::vector<double>>(*other.data);
    
    return *this;
}

size_t Matrix::get_rows_count() const { return rows; }
size_t Matrix::get_cols_count() const { return cols; }

Matrix Matrix::transpose() const{
    Matrix result(cols, rows);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result(c, r) = (r, c);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const{
    if (this->cols != other.rows) {
        std::string message = std::format("Matrix dimensions do not match for multiplication. First dims {}x{}. Seconds dims {}x{}", 
            this->rows, this->cols, other.rows, other.cols);
        throw std::invalid_argument(message);
    }

    Matrix result(this->rows, other.cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < this->cols; ++k) {
                result(i, j) += (i, k) * other(k, j);
            }
        }
    }

    return result;
}

Vector Matrix::operator*(const Vector& other) const{
    if (other.get_size() != cols) {
        throw std::invalid_argument("Matrix-Vector multiplication size mismatch!");
    }

    Vector result(rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i) += (i, j) * other(j);
        }
    }

    return result;
}

Matrix Matrix::operator*(const double& scalar) const{
    Matrix result(rows, cols);

    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            result(r, c) = (r, c) * scalar;
        }
    }

    return result;
}

Matrix& Matrix::operator-=(const Matrix& other){
    if (this->rows != other.rows || this->cols != other.cols) throw std::invalid_argument("Matrix sizes must be equal!");

    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            (r, c) -= other(r, c);
        }
    }

    return *this;
}

Matrix& Matrix::operator+=(const Matrix& other){
    if (this->rows != other.rows || this->cols != other.cols) throw std::invalid_argument("Matrix sizes must be equal!");

    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            (r, c) += other(r, c);
        }
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const{
    if (this->rows != other.rows || this->cols != other.cols){
        throw std::runtime_error("Matrix sizes must be the same!");
    }

    Matrix result(rows, cols);

    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            result(r, c) = (r, c) - other(r, c);
        }
    }
    
    return result;
}

Matrix Matrix::elementwise_mult(const Matrix& other) const{
    if (this->get_rows_count() != other.get_rows_count() || this->get_cols_count() != other.get_cols_count()){
        throw std::invalid_argument("Matrix sizes must be the same!");
    }

    Matrix result(rows, cols);
    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            result(r, c) = (r, c) * other(r, c);
        }
    }

    return result;
}

void Matrix::validate_indices(size_t r, size_t c) const {
    if (r >= rows || c >= cols) {
        throw std::out_of_range("Index out of range!");
    }
}

double Matrix::operator()(size_t r, size_t c) const{
    validate_indices(r, c);
    return (*data)[r * cols + c];
}

double& Matrix::operator()(size_t r, size_t c){
    validate_indices(r, c);
    return (*data)[r * cols + c];
}

Matrix Matrix::apply_to_elements(std::function<double(double)> func) const{
    Matrix new_matrix(this->rows, this->cols);
    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            new_matrix(r, c) = func((r, c)); 
        }
    }
    return new_matrix;
}

Vector Matrix::averaged_vector() const{
    Vector result(cols);

    for (size_t c = 0; c < cols; c++){
        
        double sum(0);
        for (size_t r = 0; r < rows; r++){
            sum += (r, c);
        }   
        result(c) = sum / rows;
    }

    return result;
}

Matrix Matrix::elementwise_add(const Vector& other) const{
    Matrix new_m(this->rows, this->cols);
    if (this->get_cols_count() != other.get_size()){
        throw std::runtime_error("Sizes must match");
    }

    for (size_t r = 0; r < this->rows; r++){
        for (size_t c = 0; c < this->cols; c++){
            new_m(r, c) = (r, c) + other(c);
        }
    }

    return new_m;
}

Vector Matrix::collapse_rows(){
    Vector result(cols);

    for(size_t c = 0; c < cols; c++){
        double sum(0);
        for (size_t r = 0; r < rows; r++){
            sum += (r, c);
        }
        result(c) = sum;
    }

    return result;
}

void Matrix::print_matrix() const{

    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++){
            std::cout << (r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

}