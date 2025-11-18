#ifndef HAPPY_PHONE_LLM_C_API_H
#define HAPPY_PHONE_LLM_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer for the LLM instance
typedef void* LLM_Handle;

// Initialize the LLM engine
void llm_init();

// Create a new LLM instance
LLM_Handle llm_create();

// Load a model into the LLM instance
bool llm_load_model(LLM_Handle handle, const char* filepath);

// Generate text using the LLM instance
const char* llm_generate(LLM_Handle handle, const char* prompt, int max_tokens);

// Destroy the LLM instance
void llm_destroy(LLM_Handle handle);

#ifdef __cplusplus
}
#endif

#endif // HAPPY_PHONE_LLM_C_API_H
