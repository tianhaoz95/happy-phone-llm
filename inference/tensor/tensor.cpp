#include "tensor.h"
#include <numeric>
#include <stdexcept>

namespace tensor {

uint64_t Tensor::get_element_size(DataType dtype) {
    switch (dtype) {
        case F32: return sizeof(float);
        case I32: return sizeof(int32_t);
        default: throw std::runtime_error("Unsupported data type");
    }
}

Tensor::Tensor(const std::vector<uint64_t>& shape, DataType dtype)
    : m_shape(shape), m_dtype(dtype) {
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t data_size = m_size * get_element_size(m_dtype);
    m_data = std::make_unique<uint8_t[]>(data_size);
}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor&& other) noexcept
    : m_shape(std::move(other.m_shape)),
      m_dtype(other.m_dtype),
      m_size(other.m_size),
      m_data(std::move(other.m_data)) {
    other.m_size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        m_shape = std::move(other.m_shape);
        m_dtype = other.m_dtype;
        m_size = other.m_size;
        m_data = std::move(other.m_data);
        other.m_size = 0;
    }
    return *this;
}

} // namespace tensor
