import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

import 'src/happy_phone_llm_bindings.dart';

class HappyPhoneLlm {
  static final HappyPhoneLlm _instance = HappyPhoneLlm._internal();
  factory HappyPhoneLlm() => _instance;

  // Flag to indicate if the class is being instantiated in a test environment.
  // This prevents the native library from being loaded during unit tests.
  static bool _isTestMode = false;

  // Public setter for _isTestMode, primarily for testing purposes.
  static void setTestMode(bool value) {
    _isTestMode = value;
  }

  HappyPhoneLlm._internal() {
    // Initialize the C++ engine only if not in test mode.
    if (!_isTestMode) {
      llmInit();
    }
  }

  Pointer<Void>? _llmHandle;

  /// Creates a new LLM instance in the C++ backend.
  void createLlm() {
    if (_isTestMode) {
      // In test mode, we don't interact with the native library.
      // We can simulate a successful creation or skip this.
      _llmHandle = Pointer.fromAddress(1); // Dummy handle for testing
      return;
    }
    _llmHandle = llmCreate();
    if (_llmHandle == nullptr) {
      throw Exception('Failed to create LLM instance.');
    }
  }

  /// Loads a model into the LLM instance.
  ///
  /// [filepath] is the path to the GGUF model file.
  bool loadModel(String filepath) {
    if (_isTestMode) {
      return true; // Simulate success in test mode
    }
    if (_llmHandle == nullptr) {
      throw Exception('LLM instance not created. Call createLlm() first.');
    }
    return llmLoadModel(_llmHandle!, filepath);
  }

  /// Generates text using the LLM instance.
  ///
  /// [prompt] is the input prompt for the LLM.
  /// [maxTokens] is the maximum number of tokens to generate.
  String generate(String prompt, int maxTokens) {
    if (_isTestMode) {
      return 'Test generated text for: $prompt'; // Simulate generation in test mode
    }
    if (_llmHandle == nullptr) {
      throw Exception('LLM instance not created. Call createLlm() first.');
    }
    return llmGenerate(_llmHandle!, prompt, maxTokens);
  }

  /// Destroys the LLM instance in the C++ backend, freeing resources.
  void destroyLlm() {
    if (_isTestMode) {
      _llmHandle = null; // Simulate destruction in test mode
      return;
    }
    if (_llmHandle != nullptr) {
      llmDestroy(_llmHandle!);
      _llmHandle = nullptr;
    }
  }
}