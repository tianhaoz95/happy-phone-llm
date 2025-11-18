## Research Report: Building a Mobile LLM Framework with Flutter

This report outlines the plan for building `happy-phone-llm`, a mobile-first LLM framework with Flutter bindings.

### 1. Functionalities

The framework will provide the following functionalities:

*   **High-performance LLM Inference:** Optimized for ARM64 architecture, leveraging CPU-based inference with the potential for future GPU/NPU acceleration.
*   **Speculative Decoding:** To accelerate inference speed.
*   **GGUF Model Support:** For efficient model loading and inference.
*   **Cross-platform:** Support for Android and iOS.
*   **Flutter API:** A simple and intuitive Dart API for interacting with the LLM backend.
*   **Model Management:** Python scripts for model training, finetuning, and conversion to GGUF format.

### 2. Technical Stack Choices

*   **Inference Engine & Kernels:** C++ for performance and portability, leveraging SIMD instructions for acceleration. We will follow the design principles of `llama.cpp` and `cactus`.
*   **Flutter Bindings:** `dart:ffi` for interoperability between Dart and C++. We will use `ffigen` to automatically generate Dart bindings from C++ headers.
*   **Model Management:** Python with libraries like `torch`, `sentencepiece`, and `gguf`.
*   **Build System:** CMake for building the C++ code for Android and iOS.

### 3. Roadmap

The development will be divided into the following phases:

**Phase 1: Core Inference Engine (2-3 weeks)**

*   Implement the core C++ inference engine in the `inference/engine` directory.
*   Implement essential kernels (e.g., matrix multiplication, attention) in the `inference/kernel` directory, optimized for ARM NEON.
*   Implement GGUF model loading and parsing.
*   Create a command-line interface for testing the C++ engine.

**Phase 2: Flutter Bindings (1-2 weeks)**

*   Set up the C++ build system for Android (NDK) and iOS (Xcode).
*   Create a C API wrapper for the C++ inference engine.
*   Use `ffigen` to generate Dart bindings from the C API.
*   Implement the Dart API in `happy_phone_llm_flutter`.

**Phase 3: Example App & Integration (1 week)**

*   Integrate the `happy_phone_llm_flutter` plugin into the `example_app`.
*   Build a simple UI for text generation to test the end-to-end functionality.
*   Add the `happy_phone_llm_flutter` as a dependency in `example_app/pubspec.yaml`:
    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      happy_phone_llm_flutter:
        path: ../happy_phone_llm_flutter
    ```

**Phase 4: Model Management Scripts (1-2 weeks)**

*   Implement a Python script in the `model` directory to convert Hugging Face models to GGUF format.
*   (Optional) Implement scripts for model training and finetuning.

**Phase 5: Advanced Features & Optimization (2-3 weeks)**

*   Implement speculative decoding in the C++ inference engine.
*   Further optimize kernels and memory management.
*   Add support for more models and architectures.

### 4. Effort Estimate

*   **Total Estimated Time:** 7-11 weeks
*   **Team Size:** 1-2 engineers with expertise in C++, Dart/Flutter, and Python.

### 5. Todo Tasks

**C++ (inference/):**

*   `[ ]` Set up CMake build system.
*   `[ ]` Implement GGUF file format loader.
*   `[ ]` Implement tensor library and operations.
*   `[ ]` Implement core LLM layers (Attention, FFN, etc.).
*   `[ ]` Implement the main inference pipeline.
*   `[ ]` Create a C API for the inference engine.
*   `[ ]` Write unit tests for the C++ code.

**Flutter (happy_phone_llm_flutter/):**

*   `[ ]` Configure `ffigen` to generate Dart bindings.
*   `[ ]` Implement the main Dart API class.
*   `[ ]` Handle asynchronous communication with the C++ backend.
*   `[ ]` Add platform-specific configurations for Android and iOS.

**Python (model/):**

*   `[ ]` Set up a Python environment with necessary dependencies.
*   `[ ]` Implement the model conversion script.
*   `[ ]` Add support for different model architectures.

This detailed plan provides a clear path forward for building the `happy-phone-llm` framework. By leveraging the strengths of C++ for performance and Flutter for cross-platform UI, we can create a powerful and easy-to-use mobile LLM solution.
