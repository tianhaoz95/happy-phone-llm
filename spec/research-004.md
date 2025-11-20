## Research Report: Next Steps - Implementing Core LLM Inference in C++

### 1. Overview

This report details the next steps for the `happy-phone-llm` framework, focusing on transforming the placeholder C++ inference engine into a functional LLM inference solution. The primary goal is to resolve the `std::bad_alloc` issues, implement robust GGUF tensor loading, integrate a proper tokenizer, and build out the core LLM inference logic.

### 2. Design

#### 2.1 Addressing `std::bad_alloc` and Robust Tensor Loading

The recurring `std::bad_alloc` indicates that the current `engine.cpp` is not handling memory efficiently for LLM tensors.

*   **Objective:** Implement memory-efficient GGUF tensor loading and management.
*   **Design Choices:**
    *   **Memory Mapping (mmap):** Investigate and implement memory-mapped I/O for loading GGUF tensors. This allows models larger than physical RAM to be loaded by leveraging virtual memory, and avoids loading the entire model into RAM at once. Reference `llama.cpp`'s approach for mmap.
    *   **Quantization Support:** Plan for supporting quantized tensor types (e.g., Q4_0, Q8_0) during loading and inference to further reduce memory footprint and improve performance. This requires extending `gguf_type_to_tensor_dtype` and the `tensor` library.
    *   **Delayed Tensor Initialization:** Only allocate memory for tensors when they are actually needed in the computation graph, rather than all at once during model load.
    *   **Review `gguf::GGUFReader`:** Ensure that the `gguf::GGUFReader` correctly parses all tensor information (dimensions, type, offset) without calculation errors that could lead to oversized memory requests.

#### 2.2 Implementing a Functional Tokenizer

The current tokenizer in `engine.cpp` is a dummy and the `GGUFReader` does not correctly parse array metadata.

*   **Objective:** Implement a proper GGUF-based tokenizer to encode prompts and decode generated tokens.
*   **Design Choices:**
    *   **GGUF Metadata Parsing for Arrays:** Extend `gguf::GGUFReader` (in `inference/gguf/gguf.h` and `gguf.cpp`) to correctly read GGUF array metadata, specifically `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, and `tokenizer.ggml.token_type`. This might involve changing `m_metadata` to handle different types of values (e.g., using `std::variant` or a specialized metadata struct).
    *   **Tokenizer Implementation:** Based on the loaded metadata, implement a BPE (Byte Pair Encoding) or SentencePiece tokenizer (depending on the model's tokenizer type) in `inference/engine/engine.cpp` to convert strings to token IDs and vice-versa.
    *   **Special Tokens:** Correctly handle special tokens like `<s>`, `</s>`, `<unk>`, `<pad>`.

#### 2.3 Implementing Core LLM Layers

The placeholders in `LLM::generate` need to be replaced with actual LLM computation.

*   **Objective:** Implement the core layers of a Transformer-based LLM.
*   **Design Choices:**
    *   **RMSNorm:** Implement a correct RMSNorm layer within `inference/kernel`.
    *   **Multi-head Attention:** Implement Multi-head Attention, including Rotary Position Embeddings (RoPE) and Key-Value (KV) caching.
        *   **KV Cache:** Manage the KV cache for efficient sequence generation.
    *   **Feed-Forward Network (FFN):** Implement the FFN layers using the loaded model weights.
    *   **Activation Functions:** Implement common activation functions like SiLU (Swish-gated Linear Unit) or GELU.
    *   **Optimized Kernels:** Continue to leverage and optimize kernels in `inference/kernel` for ARM NEON where possible.

#### 2.4 Main Inference Pipeline Refinement

The `LLM::generate` function needs to orchestrate the layers.

*   **Objective:** Build a robust and efficient LLM generation loop.
*   **Design Choices:**
    *   **Pre-fill and Decode:** Implement a pre-fill step for the initial prompt and a token-by-token decode loop.
    *   **Sampling Strategy:** Implement basic sampling (e.g., argmax, top-k, top-p) for token generation.
    *   **Batching:** Consider initial support for batching, even if small, for efficiency.

### 3. Roadmap (Next Phase)

*   **Week 1-2: Memory & Tokenizer Foundation**
    *   Implement memory mapping for `gguf::GGUFReader`.
    *   Enhance `gguf::GGUFReader` to correctly parse array metadata for tokenizer.
    *   Implement basic tokenizer (encode/decode) based on GGUF metadata.
    *   Re-enable actual model loading in `Model::load` and verify small GGUF models load without `std::bad_alloc`.
*   **Week 3-4: Core Layer Implementation**
    *   Implement RMSNorm.
    *   Implement Multi-head Attention with RoPE and KV cache.
    *   Implement Feed-Forward Network and activation functions.
*   **Week 5-6: Inference Loop & Testing**
    *   Integrate core layers into the `LLM::generate` loop.
    *   Implement basic sampling (argmax).
    *   Thoroughly test the C++ CLI with small, functional GGUF models (e.g., the dummy model).
    *   Re-integrate with Flutter example app and verify functional text generation with small models.

### 4. Effort Estimate

*   **Estimated Time:** 6-8 weeks (for a single experienced engineer)
*   **Key Skill Sets:** Advanced C++, low-level memory management, deep learning model architectures, ARM NEON intrinsics (desirable).

### 5. Todo Tasks

**C++ (inference/):**

*   `[ ]` Implement memory mapping for `gguf::GGUFReader` for efficient tensor loading.
*   `[ ]` Extend `gguf::GGUFReader` to correctly parse array metadata (tokenizer tokens).
*   `[ ]` Implement a functional tokenizer (encode/decode) using GGUF metadata in `engine.cpp`.
*   `[x]` Implement the RMSNorm layer in `inference/kernel`.
*   `[x]` Implement Multi-head Attention with RoPE and KV cache in `inference/kernel`.
*   `[x]` Implement Feed-Forward Network and activation functions in `inference/kernel`.
*   `[ ]` Refine the `LLM::generate` loop to perform actual LLM inference using implemented layers.
*   `[ ]` Implement a basic sampling strategy (e.g., argmax).
*   `[ ]` Re-enable full model loading in `Model::load` and remove all previous workarounds.
*   `[ ]` Create specific unit/integration tests for each new C++ layer and the overall inference pipeline.