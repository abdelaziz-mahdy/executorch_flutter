/// Tests for TensorType <-> ExecuTorch native type conversion.
///
/// Verifies that all 13 TensorType values map 1:1 to ExecuTorch's ETDType
/// enum values and that round-trip conversions are lossless.
library;

import 'package:executorch_dart/src/executorch_manager_base.dart';
import 'package:executorch_dart/src/types.dart';
import 'package:test/test.dart';

void main() {
  group('TensorType executorchValue', () {
    test('all types have correct ExecuTorch enum values', () {
      // Verify the enum indices match ExecuTorch's ETDType values.
      expect(TensorType.float32.executorchValue, 0,
          reason: 'float32 should map to ET_DTYPE_FLOAT32 (0)');
      expect(TensorType.float64.executorchValue, 1,
          reason: 'float64 should map to ET_DTYPE_FLOAT64 (1)');
      expect(TensorType.int64.executorchValue, 2,
          reason: 'int64 should map to ET_DTYPE_INT64 (2)');
      expect(TensorType.int32.executorchValue, 3,
          reason: 'int32 should map to ET_DTYPE_INT32 (3)');
      expect(TensorType.int16.executorchValue, 4,
          reason: 'int16 should map to ET_DTYPE_INT16 (4)');
      expect(TensorType.int8.executorchValue, 5,
          reason: 'int8 should map to ET_DTYPE_INT8 (5)');
      expect(TensorType.uint8.executorchValue, 6,
          reason: 'uint8 should map to ET_DTYPE_UINT8 (6)');
      expect(TensorType.bool_.executorchValue, 7,
          reason: 'bool_ should map to ET_DTYPE_BOOL (7)');
      expect(TensorType.uint16.executorchValue, 8,
          reason: 'uint16 should map to ET_DTYPE_UINT16 (8)');
      expect(TensorType.uint32.executorchValue, 9,
          reason: 'uint32 should map to ET_DTYPE_UINT32 (9)');
      expect(TensorType.uint64.executorchValue, 10,
          reason: 'uint64 should map to ET_DTYPE_UINT64 (10)');
      expect(TensorType.float16.executorchValue, 11,
          reason: 'float16 should map to ET_DTYPE_FLOAT16 (11)');
      expect(TensorType.bfloat16.executorchValue, 12,
          reason: 'bfloat16 should map to ET_DTYPE_BFLOAT16 (12)');
    });

    test('executorchValue equals index for all types', () {
      // Safety check: executorchValue should always equal enum index.
      for (final type in TensorType.values) {
        expect(type.executorchValue, type.index);
      }
    });
  });

  group('TensorType.fromExecuTorchValue', () {
    test('round-trips all types without loss', () {
      // The critical test: TensorType -> int -> TensorType must be identical.
      for (final original in TensorType.values) {
        final converted =
            TensorType.fromExecuTorchValue(original.executorchValue);
        expect(converted, original,
            reason: '$original should round-trip losslessly');
      }
    });

    test('fromExecuTorchValue maps all valid values', () {
      expect(TensorType.fromExecuTorchValue(0), TensorType.float32);
      expect(TensorType.fromExecuTorchValue(1), TensorType.float64);
      expect(TensorType.fromExecuTorchValue(2), TensorType.int64);
      expect(TensorType.fromExecuTorchValue(3), TensorType.int32);
      expect(TensorType.fromExecuTorchValue(4), TensorType.int16);
      expect(TensorType.fromExecuTorchValue(5), TensorType.int8);
      expect(TensorType.fromExecuTorchValue(6), TensorType.uint8);
      expect(TensorType.fromExecuTorchValue(7), TensorType.bool_);
      expect(TensorType.fromExecuTorchValue(8), TensorType.uint16);
      expect(TensorType.fromExecuTorchValue(9), TensorType.uint32);
      expect(TensorType.fromExecuTorchValue(10), TensorType.uint64);
      expect(TensorType.fromExecuTorchValue(11), TensorType.float16);
      expect(TensorType.fromExecuTorchValue(12), TensorType.bfloat16);
    });

    test('fromExecuTorchValue rejects out-of-range values', () {
      expect(
        () => TensorType.fromExecuTorchValue(-1),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => TensorType.fromExecuTorchValue(13),
        throwsA(isA<ArgumentError>()),
      );
      expect(
        () => TensorType.fromExecuTorchValue(100),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('TensorType completeness', () {
    test('has exactly 13 types (no fallbacks needed)', () {
      expect(TensorType.values.length, 13,
          reason: 'TensorType must support all 13 ExecuTorch types '
              'without fallback');
    });

    test('all types are directly supported', () {
      // No type should require a "fallback" mapping.
      // This test ensures every type maps to itself via the conversion.
      final types = [
        TensorType.float32,
        TensorType.float64,
        TensorType.int64,
        TensorType.int32,
        TensorType.int16,
        TensorType.int8,
        TensorType.uint8,
        TensorType.bool_,
        TensorType.uint16,
        TensorType.uint32,
        TensorType.uint64,
        TensorType.float16,
        TensorType.bfloat16,
      ];

      for (final type in types) {
        final roundTripped =
            TensorType.fromExecuTorchValue(type.executorchValue);
        expect(roundTripped, equals(type),
            reason: '$type must map directly, not via fallback');
      }
    });
  });

  group('TensorType sizeInBytes', () {
    test('all types have correct element size', () {
      expect(TensorType.float32.sizeInBytes, 4);
      expect(TensorType.float64.sizeInBytes, 8);
      expect(TensorType.int64.sizeInBytes, 8);
      expect(TensorType.int32.sizeInBytes, 4);
      expect(TensorType.int16.sizeInBytes, 2);
      expect(TensorType.int8.sizeInBytes, 1);
      expect(TensorType.uint8.sizeInBytes, 1);
      expect(TensorType.bool_.sizeInBytes, 1);
      expect(TensorType.uint16.sizeInBytes, 2);
      expect(TensorType.uint32.sizeInBytes, 4);
      expect(TensorType.uint64.sizeInBytes, 8);
      expect(TensorType.float16.sizeInBytes, 2);
      expect(TensorType.bfloat16.sizeInBytes, 2);
    });
  });

  group('TensorType displayName', () {
    test('all types have a non-empty display name', () {
      for (final type in TensorType.values) {
        expect(type.displayName, isNotEmpty,
            reason: '$type should have a display name');
      }
    });
  });

  // -----------------------------------------------------------------------
  // Integer encoding: verify convertNumericDataToBytes produces the exact
  // little-endian two's-complement byte layout the native layer expects.
  // -----------------------------------------------------------------------

  group('convertNumericDataToBytes integer encoding', () {
    test("int8 uses two's complement (no bias)", () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-128, -1, 0, 1, 127],
        TensorType.int8,
      );
      expect(bytes, [0x80, 0xFF, 0x00, 0x01, 0x7F]);
      // Round-trip back through a signed view.
      expect(bytes.buffer.asInt8List(), [-128, -1, 0, 1, 127]);
    });

    test('int16 round-trips values including negatives', () {
      final values = [-32768, -1, 0, 1, 32767];
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        values,
        TensorType.int16,
      );
      expect(bytes, hasLength(values.length * 2));
      expect(bytes.buffer.asInt16List(), values);
    });

    test('int32 round-trips values including negatives', () {
      final values = [-2147483648, -1, 0, 1, 2147483647];
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        values,
        TensorType.int32,
      );
      expect(bytes, hasLength(values.length * 4));
      expect(bytes.buffer.asInt32List(), values);
    });

    test('int64 round-trips values including extremes', () {
      final values = [-9223372036854775808, -1, 0, 1, 9223372036854775807];
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        values,
        TensorType.int64,
      );
      expect(bytes, hasLength(values.length * 8));
      expect(bytes.buffer.asInt64List(), values);
    });

    test('uint8 clamps and round-trips', () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-5, 0, 1, 255, 300],
        TensorType.uint8,
      );
      expect(bytes, [0, 0, 1, 255, 255]);
    });

    test('uint16 clamps and round-trips', () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-5, 0, 1, 65535, 70000],
        TensorType.uint16,
      );
      expect(bytes, hasLength(10));
      expect(bytes.buffer.asUint16List(), [0, 0, 1, 65535, 65535]);
    });

    test('uint32 clamps and round-trips', () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-5, 0, 1, 0xFFFFFFFF, 0x100000000],
        TensorType.uint32,
      );
      expect(bytes, hasLength(20));
      expect(
        bytes.buffer.asUint32List(),
        [0, 0, 1, 0xFFFFFFFF, 0xFFFFFFFF],
      );
    });

    test('uint64 encodes without crashing and clamps negatives to zero', () {
      // Dart ints are 64-bit signed, so the representable uint64 range here
      // is capped at 2^63-1. This test guards the historical clamp crash
      // (clamp(0, 0xFFFFFFFFFFFFFFFF) threw because the literal is -1).
      final values = [-5, 0, 1, 9223372036854775807];
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        values,
        TensorType.uint64,
      );
      expect(bytes, hasLength(values.length * 8));
      expect(
        bytes.buffer.asUint64List(),
        [0, 0, 1, 9223372036854775807],
      );
    });

    test('bool encodes non-zero as 1 and zero as 0', () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [0, 1, -3, 2.5],
        TensorType.bool_,
      );
      expect(bytes, [0, 1, 1, 1]);
    });
  });
}
