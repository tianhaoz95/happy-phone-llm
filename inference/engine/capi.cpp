#include "capi.h"
#include "engine.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring> // For std::strcpy

// Global storage for generated strings to ensure their lifetime
static std::vector<std::string> g_generated_strings;

void llm_init() {
    happy_phone_llm_engine_init();
}

LLM_Handle llm_create() {
    engine::LLM* llm = new engine::LLM();
    return static_cast<LLM_Handle>(llm);
}

bool llm_load_model(LLM_Handle handle, const char* filepath) {
    engine::LLM* llm = static_cast<engine::LLM*>(handle);
    if (!llm) {
        std::cerr << "Error: Invalid LLM handle." << std::endl;
        return false;
    }
    return llm->load_model(filepath);
}

const char* llm_generate(LLM_Handle handle, const char* prompt, int max_tokens) {
    engine::LLM* llm = static_cast<engine::LLM*>(handle);
    if (!llm) {
        std::cerr << "Error: Invalid LLM handle." << std::endl;
        return nullptr;
    }
    std::string result = llm->generate(prompt, max_tokens);

    // Store the generated string and return a pointer to its C-string
    g_generated_strings.push_back(result);
    return g_generated_strings.back().c_str();
}

void llm_destroy(LLM_Handle handle) {
    engine::LLM* llm = static_cast<engine::LLM*>(handle);
    if (llm) {
        delete llm;
    }
    // Clear generated strings to free memory
    g_generated_strings.clear();
}
