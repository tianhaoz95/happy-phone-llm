#include "ffn.h"
#include "matrix_multiplication.h" // For matmul
#include <iostream> // For error messages
#include <cmath>    // For std::exp
#include <stdexcept> // For std::runtime_error

namespace happy_phone_llm {
namespace kernel {

void silu(tensor::Tensor& input) {
    if (input.dtype() != tensor::F32) {
        throw std::runtime_error("SiLU only supports F32 tensors.");
    }

    float* data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.size(); ++i) {
        data[i] = data[i] * (1.0f / (1.0f + std::exp(-data[i])));
    }
}

void ffn(tensor::Tensor& output, const tensor::Tensor& input, const tensor::Tensor& w1, const tensor::Tensor& w2, const tensor::Tensor& w3) {
    // Input and Parameter Validation
    if (input.dtype() != tensor::F32 || w1.dtype() != tensor::F32 || w2.dtype() != tensor::F32 ||
        w3.dtype() != tensor::F32 || output.dtype() != tensor::F32) {
        throw std::runtime_error("FFN: All tensors must be of type F32.");
    }

    if (input.shape().size() != 2 || input.shape()[0] != 1) {
        throw std::runtime_error("FFN: Input tensor must be 1xN_EMBD.");
    }
    // Expected shapes:
    // input: [1, n_embd]
    // w1: [n_embd, intermediate_size]
    // w3: [n_embd, intermediate_size]
    // w2: [intermediate_size, n_embd]
    // output: [1, n_embd]

    uint64_t n_embd = input.shape()[1];
    uint64_t intermediate_size = w2.shape()[0];

    if (w1.shape()[0] != n_embd || w3.shape()[0] != n_embd || w2.shape()[1] != n_embd ||
        w1.shape()[1] != w3.shape()[1] || w2.shape()[0] != intermediate_size) {
        throw std::runtime_error("FFN: Mismatched tensor shapes for SwiGLU operation.");
    }
    if (output.shape()[0] != 1 || output.shape()[1] != n_embd) {
        throw std::runtime_error("FFN: Output tensor shape is incorrect.");
    }

    // 1. Linear projection for gate: gate_proj = input @ w1
    tensor::Tensor gate_proj({1, intermediate_size}, tensor::F32);
    kernel::matmul(input, w1, gate_proj);

    // 2. Apply SiLU: activated_gate = SiLU(gate_proj)
    kernel::silu(gate_proj);

    // 3. Linear projection for hidden: hidden_proj = input @ w3
    tensor::Tensor hidden_proj({1, intermediate_size}, tensor::F32);
    kernel::matmul(input, w3, hidden_proj);

    // 4. Element-wise multiplication: intermediate = hidden_proj * activated_gate
    tensor::Tensor intermediate({1, intermediate_size}, tensor::F32);
    float* intermediate_data = static_cast<float*>(intermediate.data());
    const float* hidden_data = static_cast<const float*>(hidden_proj.data());
    const float* gate_data = static_cast<const float*>(gate_proj.data());

    for (size_t i = 0; i < intermediate_size; ++i) {
        intermediate_data[i] = hidden_data[i] * gate_data[i];
    }

    // 5. Final linear projection: output = intermediate @ w2
    kernel::matmul(intermediate, w2, output);
}

} // namespace kernel
} // namespace happy_phone_llm

