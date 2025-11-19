import 'package:flutter_test/flutter_test.dart';
import 'package:happy_phone_llm_flutter/happy_phone_llm_flutter.dart';
import 'dart:io';

void main() {
  test('HappyPhoneLlm can be instantiated', () {
    // Set test mode to prevent native library loading during unit tests.
    HappyPhoneLlm.setTestMode(true);
    final llm = HappyPhoneLlm();
    expect(llm, isNotNull);
    // Reset test mode after the test.
    HappyPhoneLlm.setTestMode(false);
  });

  test('HappyPhoneLlm loads dummy model and generates text', () {
    // Ensure the dummy model exists for the test
    final dummyModelPath = 'test/test_model.gguf';
    if (!File(dummyModelPath).existsSync()) {
      fail('Dummy model file not found at $dummyModelPath. Please create it using create_dummy_gguf.py');
    }

    HappyPhoneLlm.setTestMode(false); // Enable native library loading
    final llm = HappyPhoneLlm();

    llm.createLlm();
    expect(llm.loadModel(dummyModelPath), isTrue);

    final generatedText = llm.generate("test prompt", 5);
    expect(generatedText, isNotEmpty); // Expect some output, even if dummy

    llm.destroyLlm();
  });
}
