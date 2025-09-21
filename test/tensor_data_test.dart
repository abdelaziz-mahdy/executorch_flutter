import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';

void main() {
  group('T013: TensorData validation tests', () {
    test('should create valid TensorData with required fields', () {
      // Arrange
      final shape = [1, 3, 224, 224];
      final dataType = TensorType.float32;
      final data = Uint8List.fromList(List.filled(1 * 3 * 224 * 224 * 4, 0));
      const name = 'input_tensor';

      // Act
      final tensorData = TensorData(
        shape: shape,
        dataType: dataType,
        data: data,
        name: name,
      );

      // Assert
      expect(tensorData.shape, equals(shape));
      expect(tensorData.dataType, equals(dataType));
      expect(tensorData.data, equals(data));
      expect(tensorData.name, equals(name));
    });

    test('should create valid TensorData without optional name', () {
      // Arrange
      final shape = [1, 1000];
      final dataType = TensorType.float32;
      final data = Uint8List.fromList(List.filled(1000 * 4, 0));

      // Act
      final tensorData = TensorData(
        shape: shape,
        dataType: dataType,
        data: data,
      );

      // Assert
      expect(tensorData.shape, equals(shape));
      expect(tensorData.dataType, equals(dataType));
      expect(tensorData.data, equals(data));
      expect(tensorData.name, isNull);
    });

    test('should handle different tensor types correctly', () {
      final testCases = [
        {'type': TensorType.float32, 'bytesPerElement': 4},
        {'type': TensorType.int32, 'bytesPerElement': 4},
        {'type': TensorType.int8, 'bytesPerElement': 1},
        {'type': TensorType.uint8, 'bytesPerElement': 1},
      ];

      for (final testCase in testCases) {
        // Arrange
        final shape = [2, 3];
        final type = testCase['type'] as TensorType;
        final bytesPerElement = testCase['bytesPerElement'] as int;
        final expectedDataSize = 2 * 3 * bytesPerElement;
        final data = Uint8List.fromList(List.filled(expectedDataSize, 42));

        // Act
        final tensorData = TensorData(
          shape: shape,
          dataType: type,
          data: data,
        );

        // Assert
        expect(tensorData.shape, equals(shape));
        expect(tensorData.dataType, equals(type));
        expect(tensorData.data.length, equals(expectedDataSize));
      }
    });

    test('should handle dynamic shapes with -1 dimensions', () {
      // Arrange
      final shape = [-1, 3, 224, 224];
      final dataType = TensorType.float32;
      final data = Uint8List.fromList(List.filled(3 * 224 * 224 * 4, 0));

      // Act
      final tensorData = TensorData(
        shape: shape,
        dataType: dataType,
        data: data,
      );

      // Assert
      expect(tensorData.shape, equals(shape));
      expect(tensorData.shape[0], equals(-1));
      expect(tensorData.dataType, equals(dataType));
    });

    test('should handle empty shape tensors (scalars)', () {
      // Arrange
      final shape = <int>[];
      final dataType = TensorType.float32;
      final data = Uint8List.fromList([0, 0, 0, 0]); // 4 bytes for float32

      // Act
      final tensorData = TensorData(
        shape: shape,
        dataType: dataType,
        data: data,
      );

      // Assert
      expect(tensorData.shape, isEmpty);
      expect(tensorData.dataType, equals(dataType));
      expect(tensorData.data.length, equals(4));
    });

    test('should handle large tensor data', () {
      // Arrange
      final shape = [10, 512, 512];
      final dataType = TensorType.uint8;
      final expectedSize = 10 * 512 * 512;
      final data = Uint8List.fromList(List.filled(expectedSize, 128));

      // Act
      final tensorData = TensorData(
        shape: shape,
        dataType: dataType,
        data: data,
        name: 'large_tensor',
      );

      // Assert
      expect(tensorData.shape, equals(shape));
      expect(tensorData.data.length, equals(expectedSize));
      expect(tensorData.name, equals('large_tensor'));
    });
  });
}
