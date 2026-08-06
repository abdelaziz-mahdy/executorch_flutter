import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('routed names resolve to exactly one declaration', () {
    // Referencing each name is the assertion: a name hidden from the
    // blanket export but not routed back fails to compile here.
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
}
