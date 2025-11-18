#ifndef HAPPY_PHONE_LLM_TENSOR_H
#define HAPPY_PHONE_LLM_TENSOR_H

#include <vector>
#include <cstdint>
#include <memory>

namespace tensor {

enum DataType {
    F32,
    I32,
    // Add more data types as needed
};

class Tensor {
public:
    Tensor(const std::vector<uint64_t>& shape, DataType dtype);
    ~Tensor();

    // Disable copy constructor and assignment operator for now
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor and assignment operator
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const std::vector<uint64_t>& shape() const { return m_shape; }
    DataType dtype() const { return m_dtype; }
    uint64_t size() const { return m_size; }
    void* data() { return m_data.get(); }
    const void* data() const { return m_data.get(); }

    static uint64_t get_element_size(DataType dtype);

private:
    std::vector<uint64_t> m_shape;
    DataType m_dtype;
    uint64_t m_size; // Total number of elements
    std::unique_ptr<uint8_t[]> m_data; // Raw data buffer
};

} // namespace tensor

#endif // HAPPY_PHONE_LLM_TENSOR_H