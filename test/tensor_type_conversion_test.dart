/// Tests for TensorType <-> ExecuTorch native type conversion.
///
/// Verifies that all 13 TensorType values map 1:1 to ExecuTorch's ETDType
/// enum values and that round-trip conversions are lossless.
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/types.dart';
import 'package:executorch_flutter/src/executorch_manager_base.dart';

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
          reason:
              'TensorType must support all 13 ExecuTorch types without fallback');
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
}
