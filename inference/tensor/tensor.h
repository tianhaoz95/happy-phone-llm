#ifndef HAPPY_PHONE_LLM_TENSOR_H
#define HAPPY_PHONE_LLM_TENSOR_H

#include <vector>
#include <cstdint>
#include <memory> // For std::move

namespace tensor {

enum DataType {
    F32,
    I32,
    // Add more data types as needed
};

class Tensor {
public:
    Tensor(const std::vector<uint64_t>& shape, DataType dtype);
    // New constructor for creating a view or taking ownership of external data
    Tensor(const std::vector<uint64_t>& shape, DataType dtype, void* data_ptr, bool owns_data = true);
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
    void* data() { return m_data; }
    const void* data() const { return m_data; }

    // Resize the tensor, reallocating memory if owned.
    // Data is not preserved.
    void resize(const std::vector<uint64_t>& new_shape);

    static uint64_t get_element_size(DataType dtype);

private:
    std::vector<uint64_t> m_shape;
    DataType m_dtype;
    uint64_t m_size; // Total number of elements
    uint8_t* m_data; // Raw data pointer
    bool m_owns_data; // Indicates if this tensor instance owns the data buffer

};

} // namespace tensor

#endif // HAPPY_PHONE_LLM_TENSOR_H