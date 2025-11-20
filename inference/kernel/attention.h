#ifndef HAPPY_PHONE_LLM_ATTENTION_H
#define HAPPY_PHONE_LLM_ATTENTION_H

#include "tensor.h" // Assuming tensor.h provides necessary tensor definitions

namespace happy_phone_llm {
namespace kernel {

// Declaration for multi-head attention function
void multi_head_attention(
    tensor::Tensor& output,
    const tensor::Tensor& input_tensor,
    const tensor::Tensor& wq, // Query weights
    const tensor::Tensor& wk, // Key weights
    const tensor::Tensor& wv, // Value weights
    const tensor::Tensor& wo, // Output weights
    uint32_t n_head,
    uint32_t n_embd,
    uint32_t n_rot,
    uint32_t head_dim_q, // Add head_dim_q
    uint32_t head_dim_kv, // Add head_dim_kv
    uint32_t n_kv_head, // Add n_kv_head
    uint32_t position,
    tensor::Tensor& kv_cache_k, // Key cache
    tensor::Tensor& kv_cache_v  // Value cache
);

// Declaration for softmax function
void softmax(tensor::Tensor& input);

// Declaration for Rotary Position Embeddings (RoPE) function
void rope(tensor::Tensor& input, uint32_t position, uint32_t n_rot);

} // namespace kernel
} // namespace happy_phone_llm

#endif // HAPPY_PHONE_LLM_ATTENTION_H
