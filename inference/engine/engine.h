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

private:
    Model m_model;
    // TODO: Add other inference-related members (e.g., context, tokenizer)
    // For now, a simple tokenizer placeholder
    std::map<int, std::string> m_tokenizer_id_to_token;
    std::map<std::string, int> m_tokenizer_token_to_id;
};

} // namespace engine

#endif // HAPPY_PHONE_LLM_ENGINE_H