#ifndef HAPPY_PHONE_LLM_ENGINE_H
#define HAPPY_PHONE_LLM_ENGINE_H

#include "gguf/gguf.h"
#include "tensor/tensor.h"
#include "kernel/attention.h"
#include "kernel/ffn.h"
#include "kernel/rmsnorm.h"

#include <string>
#include <memory>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

// C-style API for initialization (for Flutter FFI)
void happy_phone_llm_engine_init();

#ifdef __cplusplus
}
#endif

namespace engine {

class Model {
public:
    Model(); // Default constructor
    bool load(const std::string& filepath);

    const gguf::GGUFReader& get_gguf_reader() const { return m_gguf_reader; }
    const tensor::Tensor& get_tensor(const std::string& name) const {
        if (m_tensors.count(name)) {
            return m_tensors.at(name);
        }
        throw std::runtime_error("Tensor not found: " + name);
    }

    // Specific tensors for LLM layers (placeholders for now)
    tensor::Tensor token_embeddings;
    tensor::Tensor rms_norm_weight; // For pre-attention/pre-ffn norms
    tensor::Tensor wq, wk, wv, wo; // Attention weights
    tensor::Tensor w1, w2, w3;     // FFN weights (for SwiGLU or similar)
    tensor::Tensor output_weight;

    // Model parameters
    uint32_t n_vocab = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_layer = 0;
    uint32_t n_rot = 0; // Rotary embeddings
    float norm_eps = 1e-5; // Epsilon for RMSNorm

private:
    gguf::GGUFReader m_gguf_reader;
    // TODO: Store tensors (weights) here
    std::map<std::string, tensor::Tensor> m_tensors;
};

class LLM {
public:
    LLM();
    bool load_model(const std::string& filepath);
    std::string generate(const std::string& prompt, int max_tokens);

    // Tokenizer methods
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);

private:
    Model m_model;
    // Tokenizer members
    std::vector<std::string> m_tokenizer_id_to_token; // Use vector for direct indexing by ID
    std::map<std::string, int> m_tokenizer_token_to_id;
    std::vector<float> m_tokenizer_scores; // Token scores
    std::vector<uint32_t> m_tokenizer_token_types; // Token types
    int m_bos_token_id = -1; // Beginning of sentence token ID
    int m_eos_token_id = -1; // End of sentence token ID
    int m_unk_token_id = -1; // Unknown token ID
    int m_pad_token_id = -1; // Padding token ID

    uint32_t m_max_seq_len = 2048; // Maximum sequence length for KV cache
    // KV Cache for generation
    tensor::Tensor m_kv_cache_k;
    tensor::Tensor m_kv_cache_v;
};

} // namespace engine

#endif // HAPPY_PHONE_LLM_ENGINE_H