#include "tensor/tensor.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

void test_tensor_creation() {
    std::cout << "Running test_tensor_creation..." << std::endl;
    std::vector<uint64_t> shape = {2, 3, 4};
    tensor::Tensor t(shape, tensor::F32);

    assert(t.shape() == shape);
    assert(t.dtype() == tensor::F32);
    assert(t.size() == 2 * 3 * 4);
    assert(t.data() != nullptr);

    std::cout << "test_tensor_creation passed." << std::endl;
}

void test_tensor_data_access() {
    std::cout << "Running test_tensor_data_access..." << std::endl;
    std::vector<uint64_t> shape = {2, 2};
    tensor::Tensor t(shape, tensor::F32);

    float* data = static_cast<float*>(t.data());
    for (uint64_t i = 0; i < t.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    for (uint64_t i = 0; i < t.size(); ++i) {
        assert(data[i] == static_cast<float>(i));
    }
    std::cout << "test_tensor_data_access passed." << std::endl;
}

void test_tensor_element_size() {
    std::cout << "Running test_tensor_element_size..." << std::endl;
    assert(tensor::Tensor::get_element_size(tensor::F32) == sizeof(float));
    assert(tensor::Tensor::get_element_size(tensor::I32) == sizeof(int32_t));
    std::cout << "test_tensor_element_size passed." << std::endl;
}

int main() {
    test_tensor_creation();
    test_tensor_data_access();
    test_tensor_element_size();
    std::cout << "All tensor tests passed!" << std::endl;
    return 0;
}
