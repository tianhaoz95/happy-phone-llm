// ignore_for_file: always_specify_types
// ignore_for_file: camel_case_types
// ignore_for_file: non_constant_identifier_names
// ignore_for_file: unused_element
// ignore_for_file: unused_field
// ignore_for_file: annotate_overrides
// ignore_for_file: prefer_generic_function_type_aliases
// ignore_for_file: unused_import
// ignore_for_file: library_private_types_in_public_api

import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:ffi/src/utf8.dart';

final DynamicLibrary _dylib = () {
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open('libhappy_phone_llm_engine.dylib');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('libhappy_phone_llm_engine.so');
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('happy_phone_llm_engine.dll');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// C function `llm_init`.
typedef llm_init_C = Void Function();

/// Dart function `llm_init`.
typedef llm_init_Dart = void Function();

/// C function `llm_create`.
typedef llm_create_C = Pointer<Void> Function();

/// Dart function `llm_create`.
typedef llm_create_Dart = Pointer<Void> Function();

/// C function `llm_load_model`.
typedef llm_load_model_C = Bool Function(
  Pointer<Void> handle,
  Pointer<Char> filepath,
);

/// Dart function `llm_load_model`.
typedef llm_load_model_Dart = bool Function(
  Pointer<Void> handle,
  Pointer<Char> filepath,
);

/// C function `llm_generate`.
typedef llm_generate_C = Pointer<Char> Function(
  Pointer<Void> handle,
  Pointer<Char> prompt,
  Int32 max_tokens,
);

/// Dart function `llm_generate`.
typedef llm_generate_Dart = Pointer<Char> Function(
  Pointer<Void> handle,
  Pointer<Char> prompt,
  int max_tokens,
);

/// C function `llm_destroy`.
typedef llm_destroy_C = Void Function(
  Pointer<Void> handle,
);

/// Dart function `llm_destroy`.
typedef llm_destroy_Dart = void Function(
  Pointer<Void> handle,
);

final llm_init_Dart _llm_init = _dylib.lookupFunction<llm_init_C, llm_init_Dart>('llm_init');
final llm_create_Dart _llm_create = _dylib.lookupFunction<llm_create_C, llm_create_Dart>('llm_create');
final llm_load_model_Dart _llm_load_model = _dylib.lookupFunction<llm_load_model_C, llm_load_model_Dart>('llm_load_model');
final llm_generate_Dart _llm_generate = _dylib.lookupFunction<llm_generate_C, llm_generate_Dart>('llm_generate');
final llm_destroy_Dart _llm_destroy = _dylib.lookupFunction<llm_destroy_C, llm_destroy_Dart>('llm_destroy');

// Wrapper functions for easier use
void llmInit() => _llm_init();
Pointer<Void> llmCreate() => _llm_create();
bool llmLoadModel(Pointer<Void> handle, String filepath) => _llm_load_model(handle, filepath.toNativeUtf8().cast<Char>());
String llmGenerate(Pointer<Void> handle, String prompt, int maxTokens) {
  final promptUtf8 = prompt.toNativeUtf8();
  final resultPtr = _llm_generate(handle, promptUtf8.cast<Char>(), maxTokens);
  final result = resultPtr.cast<Utf8>().toDartString();
  // TODO: Free the C string after use if it's allocated on the C side
  malloc.free(promptUtf8); // Free the allocated native string for the prompt
  return result;
}
void llmDestroy(Pointer<Void> handle) => _llm_destroy(handle);
