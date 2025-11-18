#ifndef HAPPY_PHONE_LLM_FFN_H
#define HAPPY_PHONE_LLM_FFN_H

#include "tensor.h" // Assuming tensor.h provides necessary tensor definitions

namespace happy_phone_llm {
namespace kernel {

// Forward declaration for feed-forward network function
void ffn(tensor::Tensor& output, const tensor::Tensor& input, const tensor::Tensor& weight1, const tensor::Tensor& weight2, const tensor::Tensor& bias1, const tensor::Tensor& bias2);

} // namespace kernel
} // namespace happy_phone_llm

#endif // HAPPY_PHONE_LLM_FFN_H
