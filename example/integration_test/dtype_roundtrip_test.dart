// ignore_for_file: avoid_print, implementation_imports

/// Native dtype round-trip test.
///
/// Creates a native tensor for every TensorType and reads it back, verifying
/// the native FFI layer preserves dtype, shape, and raw bytes for all 13
/// supported dtypes (including uint16/uint32/uint64/float16/bfloat16 added
/// in native v1.3.1.9).
@Timeout(Duration(minutes: 5))
library;

import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter/src/ffi/native_tensor.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Native tensor dtype round-trip', () {
    for (final type in TensorType.values) {
      test('${type.displayName} preserves dtype, shape, and bytes', () {
        const elementCount = 6;
        final byteLength = elementCount * type.sizeInBytes;
        // Distinct byte pattern so corruption/reordering is detectable.
        final data = Uint8List.fromList(
          List<int>.generate(byteLength, (i) => (i * 7 + 3) & 0xFF),
        );

        final tensor = NativeTensor.fromTensorData(
          TensorData(
            shape: const [2, 3],
            dataType: type,
            data: data,
          ),
        );
        try {
          final back = tensor.toTensorData();
          expect(back.dataType, type,
              reason: 'dtype must survive the native round-trip');
          expect(back.shape, [2, 3]);
          expect(back.data, data,
              reason: 'raw bytes must survive the native round-trip');
        } finally {
          tensor.dispose();
        }
      });
    }

    test('rejects data size mismatched to dtype element size', () {
      // 6 elements of float16 need 12 bytes; hand 6 bytes instead.
      expect(
        () => NativeTensor.fromTensorData(
          TensorData(
            shape: const [2, 3],
            dataType: TensorType.float16,
            data: Uint8List(6),
          ),
        ),
        throwsA(isA<ExecuTorchException>()),
      );
    });
  });
}
