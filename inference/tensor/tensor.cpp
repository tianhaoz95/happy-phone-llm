#include "tensor.h"
#include <numeric>
#include <stdexcept>
#include <cstring> // For memcpy

namespace tensor {

uint64_t Tensor::get_element_size(DataType dtype) {
    switch (dtype) {
        case F32: return sizeof(float);
        case F16: return sizeof(float); // Treat F16 as float for now
        case I32: return sizeof(int32_t);
        default: throw std::runtime_error("Unsupported data type");
    }
}

// Constructor for allocating new memory (owns_data = true)
Tensor::Tensor(const std::vector<uint64_t>& shape, DataType dtype)
    : m_shape(shape), m_dtype(dtype), m_owns_data(true), m_data(nullptr) {
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t data_size = m_size * get_element_size(m_dtype);
    m_data = new uint8_t[data_size];
}

// Constructor for creating a view or taking ownership of external data
Tensor::Tensor(const std::vector<uint64_t>& shape, DataType dtype, void* data_ptr, bool owns_data)
    : m_shape(shape), m_dtype(dtype), m_owns_data(owns_data), m_data(nullptr) {
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t data_size = m_size * get_element_size(m_dtype);

    if (m_owns_data) {
        m_data = new uint8_t[data_size];
        if (data_ptr) {
            std::memcpy(m_data, data_ptr, data_size);
        }
    } else {
        m_data = static_cast<uint8_t*>(data_ptr);
    }
}

Tensor::~Tensor() {
    if (m_owns_data && m_data) {
        delete[] m_data;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : m_shape(std::move(other.m_shape)),
      m_dtype(other.m_dtype),
      m_size(other.m_size),
      m_data(other.m_data), // Transfer ownership of raw pointer
      m_owns_data(other.m_owns_data) {
    other.m_size = 0;
    other.m_data = nullptr; // Nullify other's pointer after move
    other.m_owns_data = false; // Other no longer owns data
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Deallocate current data if owned
        if (m_owns_data && m_data) {
            delete[] m_data;
        }

        m_shape = std::move(other.m_shape);
        m_dtype = other.m_dtype;
        m_size = other.m_size;
        m_data = other.m_data; // Transfer raw pointer
        m_owns_data = other.m_owns_data;

        other.m_size = 0;
        other.m_data = nullptr; // Nullify other's pointer
        other.m_owns_data = false; // Other no longer owns data
    }
    return *this;
}

void Tensor::resize(const std::vector<uint64_t>& new_shape) {
    if (!m_owns_data) {
        throw std::runtime_error("Cannot resize a tensor that does not own its data.");
    }

    // Calculate new size
    uint64_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t new_data_size = new_total_size * get_element_size(m_dtype);

    // Reallocate memory
    if (m_data) {
        delete[] m_data;
    }
    m_data = new uint8_t[new_data_size];

    // Update shape and size
    m_shape = new_shape;
    m_size = new_total_size;
}

Tensor Tensor::transpose() const {
    if (m_shape.size() != 2) {
        throw std::runtime_error("Transpose currently only supports 2D tensors.");
    }

    uint64_t rows = m_shape[0];
    uint64_t cols = m_shape[1];

    std::vector<uint64_t> transposed_shape = {cols, rows};
    Tensor transposed_tensor(transposed_shape, m_dtype);

    const float* original_data = reinterpret_cast<const float*>(m_data);
    float* transposed_data = reinterpret_cast<float*>(transposed_tensor.data());

    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            transposed_data[j * rows + i] = original_data[i * cols + j];
        }
    }

    return transposed_tensor;
}



} // namespace tensor