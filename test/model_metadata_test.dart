import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/generated/executorch_api.dart';

void main() {
  group('T014: ModelMetadata validation tests', () {
    test('should create valid ModelMetadata with all required fields', () {
      // Arrange
      const modelName = 'mobilenet_v2';
      const version = '1.0.0';
      final inputSpecs = [
        TensorSpec(
          name: 'input',
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      final outputSpecs = [
        TensorSpec(
          name: 'output',
          shape: [1, 1000],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      const estimatedMemoryMB = 14;
      final properties = <String, Object>{'framework': 'pytorch', 'quantized': false};

      // Act
      final metadata = ModelMetadata(
        modelName: modelName,
        version: version,
        inputSpecs: inputSpecs,
        outputSpecs: outputSpecs,
        estimatedMemoryMB: estimatedMemoryMB,
        properties: properties,
      );

      // Assert
      expect(metadata.modelName, equals(modelName));
      expect(metadata.version, equals(version));
      expect(metadata.inputSpecs, equals(inputSpecs));
      expect(metadata.outputSpecs, equals(outputSpecs));
      expect(metadata.estimatedMemoryMB, equals(estimatedMemoryMB));
      expect(metadata.properties, equals(properties));
    });

    test('should create valid ModelMetadata without optional properties', () {
      // Arrange
      const modelName = 'resnet50';
      const version = '2.1.0';
      final inputSpecs = [
        TensorSpec(
          name: 'input',
          shape: [1, 3, 256, 256],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      final outputSpecs = [
        TensorSpec(
          name: 'logits',
          shape: [1, 1000],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      const estimatedMemoryMB = 102;

      // Act
      final metadata = ModelMetadata(
        modelName: modelName,
        version: version,
        inputSpecs: inputSpecs,
        outputSpecs: outputSpecs,
        estimatedMemoryMB: estimatedMemoryMB,
      );

      // Assert
      expect(metadata.modelName, equals(modelName));
      expect(metadata.version, equals(version));
      expect(metadata.inputSpecs.length, equals(1));
      expect(metadata.outputSpecs.length, equals(1));
      expect(metadata.estimatedMemoryMB, equals(estimatedMemoryMB));
      expect(metadata.properties, isNull);
    });

    test('should handle multiple input and output specs', () {
      // Arrange
      const modelName = 'multi_input_output_model';
      const version = '1.0.0';
      final inputSpecs = [
        TensorSpec(
          name: 'image_input',
          shape: [1, 3, 224, 224],
          dataType: TensorType.float32,
          optional: false,
        ),
        TensorSpec(
          name: 'text_input',
          shape: [1, 512],
          dataType: TensorType.int32,
          optional: true,
        ),
      ];
      final outputSpecs = [
        TensorSpec(
          name: 'classification',
          shape: [1, 1000],
          dataType: TensorType.float32,
          optional: false,
        ),
        TensorSpec(
          name: 'embedding',
          shape: [1, 512],
          dataType: TensorType.float32,
          optional: false,
        ),
      ];
      const estimatedMemoryMB = 256;

      // Act
      final metadata = ModelMetadata(
        modelName: modelName,
        version: version,
        inputSpecs: inputSpecs,
        outputSpecs: outputSpecs,
        estimatedMemoryMB: estimatedMemoryMB,
      );

      // Assert
      expect(metadata.inputSpecs.length, equals(2));
      expect(metadata.outputSpecs.length, equals(2));
      expect(metadata.inputSpecs[0].name, equals('image_input'));
      expect(metadata.inputSpecs[1].optional, isTrue);
      expect(metadata.outputSpecs[0].name, equals('classification'));
      expect(metadata.outputSpecs[1].name, equals('embedding'));
    });

    test('should handle various property types', () {
      // Arrange
      const modelName = 'test_model';
      const version = '1.0.0';
      final inputSpecs = [
        TensorSpec(
          name: 'input',
          shape: [1, 10],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      final outputSpecs = [
        TensorSpec(
          name: 'output',
          shape: [1, 5],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      const estimatedMemoryMB = 5;
      final properties = <String, Object>{
        'framework': 'pytorch',
        'quantized': true,
        'accuracy': 0.95,
        'layers': 50,
        'training_dataset': 'imagenet',
      };

      // Act
      final metadata = ModelMetadata(
        modelName: modelName,
        version: version,
        inputSpecs: inputSpecs,
        outputSpecs: outputSpecs,
        estimatedMemoryMB: estimatedMemoryMB,
        properties: properties,
      );

      // Assert
      expect(metadata.properties?['framework'], equals('pytorch'));
      expect(metadata.properties?['quantized'], equals(true));
      expect(metadata.properties?['accuracy'], equals(0.95));
      expect(metadata.properties?['layers'], equals(50));
      expect(metadata.properties?['training_dataset'], equals('imagenet'));
    });

    test('should handle empty input/output specs lists', () {
      // Arrange
      const modelName = 'minimal_model';
      const version = '1.0.0';
      final inputSpecs = <TensorSpec>[];
      final outputSpecs = <TensorSpec>[];
      const estimatedMemoryMB = 1;

      // Act
      final metadata = ModelMetadata(
        modelName: modelName,
        version: version,
        inputSpecs: inputSpecs,
        outputSpecs: outputSpecs,
        estimatedMemoryMB: estimatedMemoryMB,
      );

      // Assert
      expect(metadata.inputSpecs, isEmpty);
      expect(metadata.outputSpecs, isEmpty);
      expect(metadata.modelName, equals(modelName));
      expect(metadata.estimatedMemoryMB, equals(estimatedMemoryMB));
    });

    test('should handle large memory estimation', () {
      // Arrange
      const modelName = 'large_model';
      const version = '1.0.0';
      final inputSpecs = [
        TensorSpec(
          name: 'input',
          shape: [1, 3, 1024, 1024],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      final outputSpecs = [
        TensorSpec(
          name: 'output',
          shape: [1, 10000],
          dataType: TensorType.float32,
          optional: false,
        )
      ];
      const estimatedMemoryMB = 8192; // 8GB

      // Act
      final metadata = ModelMetadata(
        modelName: modelName,
        version: version,
        inputSpecs: inputSpecs,
        outputSpecs: outputSpecs,
        estimatedMemoryMB: estimatedMemoryMB,
      );

      // Assert
      expect(metadata.estimatedMemoryMB, equals(8192));
      expect(metadata.modelName, equals(modelName));
    });
  });
}
