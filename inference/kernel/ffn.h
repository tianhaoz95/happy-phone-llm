#ifndef HAPPY_PHONE_LLM_FFN_H
#define HAPPY_PHONE_LLM_FFN_H

#include "tensor.h" // Assuming tensor.h provides necessary tensor definitions

namespace happy_phone_llm {
namespace kernel {

// Forward declaration for feed-forward network function (SwiGLU structure)
void ffn(tensor::Tensor& output, const tensor::Tensor& input, const tensor::Tensor& w1, const tensor::Tensor& w2, const tensor::Tensor& w3);

// Declaration for SiLU (Swish-gated Linear Unit) activation function
void silu(tensor::Tensor& input);

} // namespace kernel
} // namespace happy_phone_llm

#endif // HAPPY_PHONE_LLM_FFN_H
