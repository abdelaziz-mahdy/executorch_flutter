import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('routed names resolve to exactly one declaration', () {
    // Referencing each name is the assertion: a name hidden from the
    // blanket export but not routed back fails to compile here.
    //
    // It catches only that direction. The inverse — a name routed but not
    // hidden — is harmless on the VM and surfaces as an ambiguous-export
    // error only when compiled for web, which this test never is. Keep the
    // hide list and the routed show names in lockstep by reading them, not by
    // trusting this test.
    expect(ExecuTorchModel, isNotNull);
    expect(ExecutorchManager, isNotNull);
    expect(ExecuTorchLLM, isNotNull);
    expect(GenConfig, isNotNull);
    expect(BackendQuery, isNotNull);
    expect(ExecuTorchVersion, isNotNull);
    expect(setNativeDebugLogging, isA<Function>());
  });

  test('Flutter asset helpers are exported', () {
    expect(loadModelFromAsset, isA<Function>());
  });

  test('core types pass through the blanket export', () {
    expect(TensorData, isNotNull);
    expect(TensorType, isNotNull);
    expect(Backend, isNotNull);
    expect(executorchVersion, isNotEmpty);
  });

  test('the tokenizer reaches Flutter users on native platforms', () {
    // Tokenizer is deliberately NOT routed for web: it is dart:ffi all the
    // way down and there is no WebAssembly tokenizer to route to. It rides
    // the blanket export instead, which resolves to the core's ffi-free
    // library on web — so web code referencing Tokenizer fails to compile
    // rather than failing at runtime. That is the intended trade.
    expect(Tokenizer, isNotNull);
    expect(TokenizerFormat.huggingFace.nativeName, 'hf_json');
  });
}
