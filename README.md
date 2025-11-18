# happy-phone-llm

`happy-phone-llm` is a framework designed for running Large Language Models (LLMs) efficiently on mobile devices. It comprises a C++ inference engine optimized for ARM64 architectures, Flutter bindings for seamless mobile integration, and Python tools for model conversion and management.

## Directory Structure

-   `inference/`: Contains the core C++ LLM inference engine, including kernel implementations, GGUF parsing, and a command-line interface for testing.
-   `happy_phone_llm_flutter/`: The Flutter plugin that provides Dart FFI bindings to the C++ inference engine.
-   `example_app/`: A Flutter example application demonstrating how to use the `happy_phone_llm_flutter` plugin to load a model and generate text.
-   `model/`: Python scripts for converting Hugging Face models to the GGUF format.
-   `spec/`: Design and research documents.

## Core Inference Engine (C++)

The `inference/` directory houses the high-performance C++ inference engine.

### Building the C++ Engine

The C++ engine is built as part of the Flutter application's build process. When you build the `example_app` for Android or iOS, the C++ components are automatically compiled.

### Using the CLI for Testing C++

A command-line interface (CLI) tool is available to test the C++ inference engine directly, without the Flutter frontend.

**Prerequisites:**
-   CMake (version 3.10 or higher)
-   A C++ compiler (e.g., g++, clang)

**Steps:**

1.  **Navigate to the `inference` directory:**
    ```bash
    cd inference
    ```

2.  **Create a build directory and configure CMake:**
    ```bash
    mkdir build && cd build
    cmake ..
    ```

3.  **Build the CLI executable:**
    ```bash
    cmake --build . --target happy_phone_llm_cli
    ```

4.  **Run the CLI tool:**
    ```bash
    ./cli/happy_phone_llm_cli <path_to_your_gguf_model> "<your_prompt_text>" [max_tokens]
    ```
    -   `<path_to_your_gguf_model>`: The absolute or relative path to your GGUF model file.
    -   `"<your_prompt_text>"`: The text prompt for the LLM (enclose in quotes if it contains spaces).
    -   `[max_tokens]`: Optional. The maximum number of tokens to generate (defaults to 50).

    **Example:**
    ```bash
    ./cli/happy_phone_llm_cli ../model/model.gguf "Hello, how are you?" 100
    ```
    *(Note: The `../model/model.gguf` path is a placeholder. You will need to convert a model first using the Python script.)*

## Flutter Integration

The `happy_phone_llm_flutter` plugin provides the Dart API to interact with the C++ inference engine. The `example_app` demonstrates its usage.

### Using the `happy_phone_llm_flutter` Library

1.  **Add the dependency:**
    In your `pubspec.yaml`, add:
    ```yaml
    dependencies:
      happy_phone_llm_flutter:
        path: ../happy_phone_llm_flutter # Or a published version
    ```

2.  **Import and use in your Dart code:**
    ```dart
    import 'package:happy_phone_llm_flutter/happy_phone_llm_flutter.dart';

    // ...
    final HappyPhoneLlm _llm = HappyPhoneLlm();
    _llm.createLlm();
    _llm.loadModel("/path/to/your/model.gguf");
    String generatedText = await _llm.generate("Your prompt", 50);
    _llm.destroyLlm();
    // ...
    ```

### Testing with the Example App

The `example_app` provides a basic UI to interact with the LLM.

**Prerequisites:**
-   Flutter SDK installed and configured.
-   An Android device or emulator, or an iOS device or simulator.

**Steps:**

1.  **Navigate to the `example_app` directory:**
    ```bash
    cd example_app
    ```

2.  **Get Flutter dependencies:**
    ```bash
    flutter pub get
    ```

3.  **Run the app:**
    ```bash
    flutter run
    ```
    Or build for a specific platform:
    ```bash
    flutter build apk   # For Android
    flutter build ios   # For iOS
    ```

4.  **Interact with the UI:**
    -   Enter a prompt in the text field.
    -   Tap "Load Model" (currently uses a placeholder path `/data/local/tmp/model.gguf`).
    -   Tap "Generate Text" to see the LLM's response.
    -   Tap "Clear" to reset the input and output.

## Model Management (Python)

The `model/` directory contains tools for preparing LLM models.

### Converting Hugging Face Models to GGUF

The `convert_to_gguf.py` script allows you to convert models from the Hugging Face Hub into the GGUF format, which is compatible with the C++ inference engine.

**Prerequisites:**
-   Python 3.x
-   `torch`, `transformers`, `huggingface_hub`, `numpy` (install via `pip install -r model/requirements.txt`)

**Steps:**

1.  **Navigate to the `model` directory:**
    ```bash
    cd model
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the conversion script:**
    ```bash
    python convert_to_gguf.py <huggingface_model_name_or_path> --output_dir <output_directory> --filename <output_filename.gguf>
    ```
    -   `<huggingface_model_name_or_path>`: The name of the model on Hugging Face Hub (e.g., `gpt2`, `facebook/opt-125m`) or a local path to a downloaded model.
    -   `--output_dir`: Optional. The directory to save the GGUF file (defaults to current directory).
    -   `--filename`: Optional. The name of the output GGUF file (defaults to `model.gguf`).

    **Example:**
    ```bash
    python convert_to_gguf.py gpt2 --output_dir ../inference/model --filename gpt2.gguf
    ```
    *(Note: Ensure the output directory is accessible by your mobile application if you intend to load it on device.)*

## Future Considerations

-   **Speculative Decoding**: Integrate for significant speedups.
-   **Kernel and Memory Optimization**: Continuous profiling and optimization for various mobile devices.
-   **Broader Model Support**: Expand support for more LLM architectures and quantization schemes.
-   **Stop Generation**: Implement actual cancellation logic in C++ and Dart.
