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
    uint32_t head_dim_q, // Use head_dim_q
    uint32_t head_dim_kv, // Use head_dim_kv
    uint32_t n_kv_head,
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

    // head_dim_q and head_dim_kv validation
    if (head_dim_q == 0 || head_dim_kv == 0) {
        throw std::runtime_error("Multi-head Attention: head_dim_q and head_dim_kv cannot be zero.");
    }
    
    uint32_t total_q_output_dim = wq.shape()[1]; // The output dimension of the q_proj weight
    uint32_t total_k_output_dim = wk.shape()[1]; // The output dimension of the k_proj weight
    uint32_t total_v_output_dim = wv.shape()[1]; // The output dimension of the v_proj weight

    // 1. Linear Projections for Q, K, V
    // Input is 1 x n_embd. Weights are n_embd x (total_q/k/v_output_dim). Output is 1 x (total_q/k/v_output_dim).
    tensor::Tensor q_proj({1, total_q_output_dim}, tensor::F32);
    tensor::Tensor k_proj({1, total_k_output_dim}, tensor::F32);
    tensor::Tensor v_proj({1, total_v_output_dim}, tensor::F32);

    kernel::matmul(input_tensor, wq, q_proj);
    kernel::matmul(input_tensor, wk, k_proj);
    kernel::matmul(input_tensor, wv, v_proj);

    // 2. Reshape for Multi-head and Apply RoPE
    // Apply RoPE to each head's q and k
    for (uint32_t head = 0; head < n_head; ++head) {
        // Only apply RoPE to Q heads
        tensor::Tensor q_head_view({1, head_dim_q}, tensor::F32, static_cast<float*>(q_proj.data()) + head * head_dim_q);
        kernel::rope(q_head_view, position, n_rot);
    }
    for (uint32_t head = 0; head < n_kv_head; ++head) {
        // Only apply RoPE to K heads
        tensor::Tensor k_head_view({1, head_dim_kv}, tensor::F32, static_cast<float*>(k_proj.data()) + head * head_dim_kv);
        kernel::rope(k_head_view, position, n_rot);
    }
    // Now q_proj and k_proj contain the RoPE-applied values

    // 3. KV Cache Update and Retrieval
    // Assuming kv_cache_k and kv_cache_v are initially sized [max_seq_len, total_k_output_dim] and [max_seq_len, total_v_output_dim]
    // And we are writing to `position` row.

    // Copy k_proj (1 x total_k_output_dim) to kv_cache_k at 'position'
    std::memcpy(static_cast<float*>(kv_cache_k.data()) + position * total_k_output_dim,
                static_cast<float*>(k_proj.data()),
                total_k_output_dim * sizeof(float));

    // Copy v_proj (1 x total_v_output_dim) to kv_cache_v at 'position'
    std::memcpy(static_cast<float*>(kv_cache_v.data()) + position * total_v_output_dim,
                static_cast<float*>(v_proj.data()),
                total_v_output_dim * sizeof(float));
    
    // Create tensors for the full K and V from the cache up to current position
    // current_seq_len = position + 1
    uint32_t current_seq_len = position + 1;

    // These views point to the relevant part of the KV cache
    tensor::Tensor current_k_cache_view({current_seq_len, total_k_output_dim}, tensor::F32, static_cast<float*>(kv_cache_k.data()));
    tensor::Tensor current_v_cache_view({current_seq_len, total_v_output_dim}, tensor::F32, static_cast<float*>(kv_cache_v.data()));

    // Prepare attention output container. It should match the input dimension expected by wo.
    // Based on wo.shape()[0] = 2048, and n_head = 16, head_dim_q = 128, this means n_head * head_dim_q = 16 * 128 = 2048.
        tensor::Tensor head_outputs({1, n_head * head_dim_q}, tensor::F32); // To store output concatenated from all heads

    for (uint32_t head = 0; head < n_head; ++head) {
        // Get query head (1 x head_dim_q)
        tensor::Tensor q_head_view({1, head_dim_q}, tensor::F32, static_cast<float*>(q_proj.data()) + head * head_dim_q);

        // Create a query view for matmul that matches key/value head dimension
        tensor::Tensor q_head_view_for_matmul({1, head_dim_kv}, tensor::F32);
        std::memcpy(static_cast<float*>(q_head_view_for_matmul.data()), static_cast<float*>(q_head_view.data()), head_dim_kv * sizeof(float));

        // Get key and value for this head from the cache (current_seq_len x head_dim_kv)
        // For GQA, key and value heads are repeated if n_head > n_kv_head
        uint32_t kv_head_idx = head / (n_head / n_kv_head); // Map current head to a KV head

        tensor::Tensor k_for_attn({current_seq_len, head_dim_kv}, tensor::F32);
        tensor::Tensor v_for_attn({current_seq_len, head_dim_kv}, tensor::F32);

        for (uint32_t s = 0; s < current_seq_len; ++s) {
            std::memcpy(static_cast<float*>(k_for_attn.data()) + s * head_dim_kv,
                        static_cast<const float*>(current_k_cache_view.data()) + s * total_k_output_dim + kv_head_idx * head_dim_kv,
                        head_dim_kv * sizeof(float));
            std::memcpy(static_cast<float*>(v_for_attn.data()) + s * head_dim_kv,
                        static_cast<const float*>(current_v_cache_view.data()) + s * total_v_output_dim + kv_head_idx * head_dim_kv,
                        head_dim_kv * sizeof(float));
        }

        // 4. Self-attention Calculation
        // scores = Q @ K_T
        // Transpose k_for_attn (current_seq_len x head_dim_kv) to (head_dim_kv x current_seq_len)
        tensor::Tensor k_for_attn_T({head_dim_kv, current_seq_len}, tensor::F32);
        for(uint32_t r = 0; r < current_seq_len; ++r) {
            for(uint32_t c = 0; c < head_dim_kv; ++c) {
                static_cast<float*>(k_for_attn_T.data())[c * current_seq_len + r] = static_cast<float*>(k_for_attn.data())[r * head_dim_kv + c];
            }
        }
        
        tensor::Tensor scores({1, current_seq_len}, tensor::F32);
        kernel::matmul(q_head_view_for_matmul, k_for_attn_T, scores);

        // Scale scores
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim_kv)); // Use head_dim_kv for scaling
        for (size_t i = 0; i < scores.size(); ++i) {
            static_cast<float*>(scores.data())[i] *= scale_factor;
        }

        // Apply softmax
        kernel::softmax(scores);

        // attention_output = scores @ V
        tensor::Tensor head_output({1, head_dim_kv}, tensor::F32);
        kernel::matmul(scores, v_for_attn, head_output);

        // Store head output into the corresponding section of head_outputs
        std::memcpy(static_cast<float*>(head_outputs.data()) + head * head_dim_q, // Use head_dim_q here for indexing
                    static_cast<float*>(head_output.data()),
                    head_dim_kv * sizeof(float)); // Copy head_dim_kv data
    }

    // 5. Concatenate Heads is implicitly done as head_outputs is a flat 1 x (n_head * head_dim_q) tensor

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