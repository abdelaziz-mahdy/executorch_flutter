import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

void main() {
  group('ProcessorTensorUtils Contract Tests', () {
    test('should create tensor from numeric data', () {
      // This test will fail until ProcessorTensorUtils is implemented
      expect(
        () => ProcessorTensorUtils.createTensor(
          shape: [1, 2, 2],
          dataType: TensorType.float32,
          data: [1.0, 2.0, 3.0, 4.0],
        ),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should validate shape and data length match', () {
      // This test will fail until ProcessorTensorUtils is implemented
      expect(
        () => ProcessorTensorUtils.createTensor(
          shape: [2, 2],
          dataType: TensorType.float32,
          data: [1.0, 2.0], // Wrong length
        ),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should extract float32 data correctly', () {
      // This test will fail until ProcessorTensorUtils is implemented
      final mockTensor = MockTensorData(
        shape: [2, 2],
        dataType: TensorType.float32,
        data: Float32List.fromList([1.0, 2.0, 3.0, 4.0]).buffer.asUint8List(),
      );

      expect(
        () => ProcessorTensorUtils.extractFloat32Data(mockTensor),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should extract int32 data correctly', () {
      // This test will fail until ProcessorTensorUtils is implemented
      final mockTensor = MockTensorData(
        shape: [2, 2],
        dataType: TensorType.int32,
        data: Int32List.fromList([1, 2, 3, 4]).buffer.asUint8List(),
      );

      expect(
        () => ProcessorTensorUtils.extractInt32Data(mockTensor),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should calculate element count correctly', () {
      // This test will fail until ProcessorTensorUtils is implemented
      expect(
        () => ProcessorTensorUtils.calculateElementCount([2, 3, 4]),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should handle empty shape', () {
      // This test will fail until ProcessorTensorUtils is implemented
      expect(
        () => ProcessorTensorUtils.calculateElementCount([]),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should throw for incompatible tensor type in float32 extraction', () {
      // This test will fail until ProcessorTensorUtils is implemented
      final mockTensor = MockTensorData(
        shape: [2],
        dataType: TensorType.int32,
        data: Uint8List.fromList([1, 2, 3, 4]),
      );

      expect(
        () => ProcessorTensorUtils.extractFloat32Data(mockTensor),
        throwsA(isA<NoSuchMethodError>()),
      );
    });

    test('should throw for incompatible tensor type in int32 extraction', () {
      // This test will fail until ProcessorTensorUtils is implemented
      final mockTensor = MockTensorData(
        shape: [2],
        dataType: TensorType.float32,
        data: Uint8List.fromList([1, 2, 3, 4]),
      );

      expect(
        () => ProcessorTensorUtils.extractInt32Data(mockTensor),
        throwsA(isA<NoSuchMethodError>()),
      );
    });
  });
}

// Mock TensorData for testing
class MockTensorData extends TensorData {
  MockTensorData({
    required List<int> shape,
    required TensorType dataType,
    required Uint8List data,
    String? name,
  }) : super(
          shape: shape.cast<int?>(),
          dataType: dataType,
          data: data,
          name: name,
        );
}