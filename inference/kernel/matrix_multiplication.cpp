#include "matrix_multiplication.h"
#include <stdexcept>
#include <iostream>

namespace happy_phone_llm {
namespace kernel {

void matmul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C) {
    if (A.dtype() != tensor::F32 || B.dtype() != tensor::F32 || C.dtype() != tensor::F32) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + ": Matmul only supports F32 tensors for now.");
    }

    if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 2) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + ": Matmul only supports 2D tensors.");
    }

    uint64_t M = A.shape()[0];
    uint64_t K = A.shape()[1];
    uint64_t N = B.shape()[1];

    if (K != B.shape()[0]) { // Re-enabled
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + ": Matrix A columns must match Matrix B rows for multiplication.");
    }
    if (M != C.shape()[0] || N != C.shape()[1]) {
        throw std::runtime_error("Result matrix C dimensions do not match expected output.");
    }

    const float* a_data = static_cast<const float*>(A.data());
    const float* b_data = static_cast<const float*>(B.data());
    float* c_data = static_cast<float*>(C.data());

    for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint64_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }
}

} // namespace kernel
} // namespace happy_phone_llm
