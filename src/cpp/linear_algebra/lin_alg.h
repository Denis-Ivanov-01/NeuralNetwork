#pragma once

#include <vector>
#include <memory>
#include <functional>


namespace lin_alg{

class Matrix;

class Vector{
private:
size_t size;
std::unique_ptr<std::vector<double>> data;

void validate_index(size_t i) const;

public:
Vector(size_t size);

Vector(const Vector& other);

Vector(const std::vector<double>& other);

size_t get_size() const;

double& operator()(size_t i);

const double operator()(size_t i) const;

Vector operator*(double scalar) const;  // NEW: Ensure it's const
Vector operator*(const Matrix& other) const;
Vector operator+(const Vector& other) const;
Vector& operator=(const Vector& other); // Copy assignment
Vector& operator-=(const Vector& other); // NEW: Vector -= Vector
Vector& operator+=(const Vector& other);

Matrix transpose() const;  // Convert column vector to row matrix

static Vector from_std_vector(const std::vector<double>& v);

static Vector from_matrix_row(const Matrix& inputs, size_t row);

void print() const;

};


class Matrix{
private:
size_t rows;
size_t cols;

std::unique_ptr<std::vector<double>> data;

void validate_indices(size_t r, size_t c) const;

public:

Matrix(size_t rows, size_t cols);

Matrix(const Matrix& other);

Matrix transpose() const;

size_t get_rows_count() const;

size_t get_cols_count() const;

double operator()(size_t r, size_t c) const;

double& operator()(size_t r, size_t c);

Matrix operator*(const double& scalar) const; // NEW: Matrix * scalar
Matrix operator*(const Matrix& other) const;   // Matrix-Matrix multiplication
Vector operator*(const Vector& other) const;   // Matrix-Vector multiplication
Matrix& operator=(const Matrix& other);  // Copy assignment
Matrix& operator-=(const Matrix& other); // NEW: Matrix -= Matrix
Matrix& operator+=(const Matrix& other);
Matrix operator-(const Matrix& other) const;

Matrix elementwise_mult(const Matrix& other) const;

Matrix apply_to_elements(std::function<double(double)>) const;

/// @brief Collapses matrix into an averaged vector
/// @return A vector with averaged items in each column 
Vector averaged_vector() const;

Vector collapse_rows();

Matrix elementwise_add(const Vector& other) const;

void print_matrix() const;
};

}