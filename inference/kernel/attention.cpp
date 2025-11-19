#include "attention.h"
#include "matrix_multiplication.h" // For matmul
#include <iostream>
#include <cmath> // For exp, sqrt
#include <numeric> // For std::accumulate
#include <algorithm> // For std::max_element
#include <vector>
#include <cstring> // For memcpy

namespace happy_phone_llm {
namespace kernel {

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
    uint32_t position,
    tensor::Tensor& kv_cache_k, // Key cache
    tensor::Tensor& kv_cache_v  // Value cache
) {
    // Input and Parameter Validation
    if (input_tensor.dtype() != tensor::F32 || wq.dtype() != tensor::F32 || wk.dtype() != tensor::F32 ||
        wv.dtype() != tensor::F32 || wo.dtype() != tensor::F32 || output.dtype() != tensor::F32) {
        throw std::runtime_error("Multi-head Attention: All tensors must be of type F32.");
    }
    if (input_tensor.shape().size() != 2 || input_tensor.shape()[0] != 1 || input_tensor.shape()[1] != n_embd) {
        throw std::runtime_error("Multi-head Attention: Input tensor must be 1xN_EMBD.");
    }

    // head_dim
    uint32_t head_dim = n_embd / n_head;
    if (n_embd % n_head != 0) {
        throw std::runtime_error("Multi-head Attention: n_embd must be divisible by n_head.");
    }

    // 1. Linear Projections for Q, K, V
    // Input is 1 x n_embd. Weights are n_embd x n_embd. Output is 1 x n_embd.
    tensor::Tensor q_proj({1, n_embd}, tensor::F32);
    tensor::Tensor k_proj({1, n_embd}, tensor::F32);
    tensor::Tensor v_proj({1, n_embd}, tensor::F32);

    kernel::matmul(input_tensor, wq, q_proj);
    kernel::matmul(input_tensor, wk, k_proj);
    kernel::matmul(input_tensor, wv, v_proj);

    // 2. Reshape for Multi-head and Apply RoPE
    // Reshape from 1 x n_embd to n_head x head_dim
    // Then apply RoPE to each head's q and k
    for (uint32_t head = 0; head < n_head; ++head) {
        // Create temporary tensors that are views into the q_proj and k_proj for each head
        // Note: Tensor constructor with existing data pointer does not manage memory
        tensor::Tensor q_head_view({1, head_dim}, tensor::F32, static_cast<float*>(q_proj.data()) + head * head_dim);
        tensor::Tensor k_head_view({1, head_dim}, tensor::F32, static_cast<float*>(k_proj.data()) + head * head_dim);
        
        kernel::rope(q_head_view, position, n_rot);
        kernel::rope(k_head_view, position, n_rot);
    }
    // Now q_proj and k_proj contain the RoPE-applied values

    // 3. KV Cache Update and Retrieval
    // Assuming kv_cache_k and kv_cache_v are initially sized [max_seq_len, n_embd]
    // And we are writing to `position` row.
    // This is a simplified cache management. A real cache would be more dynamic.

    // Copy k_proj (1 x n_embd) to kv_cache_k at 'position'
    std::memcpy(static_cast<float*>(kv_cache_k.data()) + position * n_embd,
                static_cast<float*>(k_proj.data()),
                n_embd * sizeof(float));

    // Copy v_proj (1 x n_embd) to kv_cache_v at 'position'
    std::memcpy(static_cast<float*>(kv_cache_v.data()) + position * n_embd,
                static_cast<float*>(v_proj.data()),
                n_embd * sizeof(float));
    
    // Create tensors for the full K and V from the cache up to current position
    // current_seq_len = position + 1
    uint32_t current_seq_len = position + 1;

    // These views point to the relevant part of the KV cache
    tensor::Tensor current_k_cache_view({current_seq_len, n_embd}, tensor::F32, static_cast<float*>(kv_cache_k.data()));
    tensor::Tensor current_v_cache_view({current_seq_len, n_embd}, tensor::F32, static_cast<float*>(kv_cache_v.data()));

    // Prepare attention output container
    tensor::Tensor head_outputs({1, n_embd}, tensor::F32); // To store output concatenated from all heads

    for (uint32_t head = 0; head < n_head; ++head) {
        // Get query head (1 x head_dim)
        tensor::Tensor q_head_view({1, head_dim}, tensor::F32, static_cast<float*>(q_proj.data()) + head * head_dim);

        // Get key and value for this head from the cache (current_seq_len x head_dim)
        // This involves creating a copy because matmul expects contiguous memory and full tensors.
        tensor::Tensor k_for_attn({current_seq_len, head_dim}, tensor::F32);
        tensor::Tensor v_for_attn({current_seq_len, head_dim}, tensor::F32);

        for (uint32_t s = 0; s < current_seq_len; ++s) {
            std::memcpy(static_cast<float*>(k_for_attn.data()) + s * head_dim,
                        static_cast<const float*>(current_k_cache_view.data()) + s * n_embd + head * head_dim,
                        head_dim * sizeof(float));
            std::memcpy(static_cast<float*>(v_for_attn.data()) + s * head_dim,
                        static_cast<const float*>(current_v_cache_view.data()) + s * n_embd + head * head_dim,
                        head_dim * sizeof(float));
        }

        // 4. Self-attention Calculation
        // scores = Q @ K_T
        // Transpose k_for_attn (current_seq_len x head_dim) to (head_dim x current_seq_len)
        tensor::Tensor k_for_attn_T({head_dim, current_seq_len}, tensor::F32);
        for(uint32_t r = 0; r < current_seq_len; ++r) {
            for(uint32_t c = 0; c < head_dim; ++c) {
                static_cast<float*>(k_for_attn_T.data())[c * current_seq_len + r] = static_cast<float*>(k_for_attn.data())[r * head_dim + c];
            }
        }
        
        tensor::Tensor scores({1, current_seq_len}, tensor::F32);
        kernel::matmul(q_head_view, k_for_attn_T, scores);

        // Scale scores
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (size_t i = 0; i < scores.size(); ++i) {
            static_cast<float*>(scores.data())[i] *= scale_factor;
        }

        // Apply softmax
        kernel::softmax(scores);

        // attention_output = scores @ V
        tensor::Tensor head_output({1, head_dim}, tensor::F32);
        kernel::matmul(scores, v_for_attn, head_output);

        // Store head output into the corresponding section of head_outputs
        std::memcpy(static_cast<float*>(head_outputs.data()) + head * head_dim,
                    static_cast<float*>(head_output.data()),
                    head_dim * sizeof(float));
    }

    // 5. Concatenate Heads is implicitly done as head_outputs is a flat 1 x n_embd tensor

    // 6. Output Projection
    kernel::matmul(head_outputs, wo, output);
}

void softmax(tensor::Tensor& input) {
    if (input.dtype() != tensor::F32) {
        throw std::runtime_error("Softmax only supports F32 tensors.");
    }

    if (input.shape().empty()) {
        return; // Nothing to do for empty tensor
    }

    const float* input_data = static_cast<const float*>(input.data());
    float* output_data = static_cast<float*>(input.data()); // Softmax operates in-place

    if (input.shape().size() == 1) { // 1D tensor
        uint64_t size = input.shape()[0];
        if (size == 0) return;

        // Find max for numerical stability
        float max_val = input_data[0];
        for (uint64_t i = 1; i < size; ++i) {
            if (input_data[i] > max_val) {
                max_val = input_data[i];
            }
        }

        double sum_exp = 0.0;
        for (uint64_t i = 0; i < size; ++i) {
            output_data[i] = std::exp(input_data[i] - max_val);
            sum_exp += output_data[i];
        }

        for (uint64_t i = 0; i < size; ++i) {
            output_data[i] = static_cast<float>(output_data[i] / sum_exp);
        }
    } else if (input.shape().size() == 2) { // 2D tensor (e.g., [batch_size, sequence_length] or [num_heads, seq_len])
        uint64_t rows = input.shape()[0];
        uint64_t cols = input.shape()[1];

        for (uint64_t r = 0; r < rows; ++r) {
            const float* row_input_data = input_data + r * cols;
            float* row_output_data = output_data + r * cols;

            // Find max for numerical stability for the current row
            float max_val = row_input_data[0];
            for (uint64_t c = 1; c < cols; ++c) {
                if (row_input_data[c] > max_val) {
                    max_val = row_input_data[c];
                }
            }

            double sum_exp = 0.0;
            for (uint64_t c = 0; c < cols; ++c) {
                row_output_data[c] = std::exp(row_input_data[c] - max_val);
                sum_exp += row_output_data[c];
            }

            for (uint64_t c = 0; c < cols; ++c) {
                row_output_data[c] = static_cast<float>(row_output_data[c] / sum_exp);
            }
        }
    } else {
        throw std::runtime_error("Softmax only supports 1D or 2D tensors for now.");
    }
}

void rope(tensor::Tensor& input, uint32_t position, uint32_t n_rot) {
    if (input.dtype() != tensor::F32) {
        throw std::runtime_error("RoPE only supports F32 tensors.");
    }
    if (input.shape().empty()) {
        return;
    }

    // RoPE is applied to the last dimension of the tensor
    uint64_t last_dim = input.shape().back();
    if (n_rot > last_dim) {
        throw std::runtime_error("RoPE: n_rot cannot be greater than the last dimension of the input tensor.");
    }

    float* data = static_cast<float*>(input.data());

    // Iterate through all "rows" or "vectors" in the tensor that need RoPE applied
    // This handles 1D, 2D, or higher-dimensional tensors, applying RoPE to the last_dim
    uint64_t num_vectors = input.size() / last_dim;

    for (uint64_t v = 0; v < num_vectors; ++v) {
        float* current_vector_data = data + v * last_dim;

        for (uint32_t i = 0; i < n_rot; i += 2) {
            float freq_cis = static_cast<float>(position) / std::pow(10000.0f, static_cast<float>(i) / n_rot);

            float cos_val = std::cos(freq_cis);
            float sin_val = std::sin(freq_cis);

            float v0 = current_vector_data[i];
            float v1 = current_vector_data[i+1];

            current_vector_data[i]   = v0 * cos_val - v1 * sin_val;
            current_vector_data[i+1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

} // namespace kernel
} // namespace happy_phone_llm