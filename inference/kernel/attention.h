#ifndef HAPPY_PHONE_LLM_ATTENTION_H
#define HAPPY_PHONE_LLM_ATTENTION_H

#include "tensor.h" // Assuming tensor.h provides necessary tensor definitions

namespace happy_phone_llm {
namespace kernel {

// Forward declaration for attention function
void attention(tensor::Tensor& output, const tensor::Tensor& query, const tensor::Tensor& key, const tensor::Tensor& value, float scale);

} // namespace kernel
} // namespace happy_phone_llm

#endif // HAPPY_PHONE_LLM_ATTENTION_H
