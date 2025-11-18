import 'package:flutter_test/flutter_test.dart';
import 'package:happy_phone_llm_flutter/happy_phone_llm_flutter.dart';

void main() {
  test('HappyPhoneLlm can be instantiated', () {
    // Set test mode to prevent native library loading during unit tests.
    HappyPhoneLlm.setTestMode(true);
    final llm = HappyPhoneLlm();
    expect(llm, isNotNull);
    // Reset test mode after the test.
    HappyPhoneLlm.setTestMode(false);
  });
}
