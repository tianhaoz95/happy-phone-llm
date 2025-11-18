#ifndef HAPPY_PHONE_LLM_ENGINE_H
#define HAPPY_PHONE_LLM_ENGINE_H

#include "gguf/gguf.h"
#include "tensor/tensor.h"

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
    bool load(const std::string& filepath);

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
};

} // namespace engine

#endif // HAPPY_PHONE_LLM_ENGINE_H