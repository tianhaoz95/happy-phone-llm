# Gemini CLI Verification

After finishing implementing a task, please ensure the correctness of the implementation by performing the following steps:

## 1. Run Unit Tests for `happy_phone_llm_flutter/`

Navigate to the `happy_phone_llm_flutter/` directory and run the unit tests:

```bash
cd happy_phone_llm_flutter/
flutter test
```

## 2. Build CLI for `inference/`

Navigate to the `inference/` directory and build the CLI application:

```bash
cd inference/
mkdir -p build
cd build
cmake ..
cmake --build . --target happy_phone_llm_cli
```

## 3. Build APK for `example_app/`

Navigate to the `example_app/` directory and build the Android APK:

```bash
cd example_app/
flutter build apk
```
