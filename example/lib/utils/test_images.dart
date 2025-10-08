/// Test image utilities for ExecuTorch model testing
library;

import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// Pre-loaded test images available in the app assets
class TestImages {
  TestImages._();

  // Classification test images
  static const String cat = 'assets/images/cat.jpg';
  static const String dog = 'assets/images/dog.jpg';
  static const String car = 'assets/images/car.jpg';
  static const String person = 'assets/images/person.jpg';

  // Object detection test images
  static const String street = 'assets/images/street.jpg';

  /// All available test images
  static const List<String> all = [cat, dog, car, person, street];

  /// Get a temporary file from an asset image
  /// This is useful for APIs that require a File path instead of asset bytes
  static Future<File> getFileFromAsset(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final tempDir = await getTemporaryDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${tempDir.path}/$fileName');
    await file.writeAsBytes(
      byteData.buffer.asUint8List(
        byteData.offsetInBytes,
        byteData.lengthInBytes,
      ),
    );
    return file;
  }

  /// Get image bytes from asset
  static Future<Uint8List> getBytesFromAsset(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    return byteData.buffer.asUint8List(
      byteData.offsetInBytes,
      byteData.lengthInBytes,
    );
  }

  /// Get a human-readable name for the test image
  static String getName(String assetPath) {
    final fileName = assetPath.split('/').last;
    final name = fileName.split('.').first;
    return name.substring(0, 1).toUpperCase() + name.substring(1);
  }

  /// Get description for the test image
  static String getDescription(String assetPath) {
    switch (assetPath) {
      case cat:
        return 'Cat image for classification testing';
      case dog:
        return 'Dog image for classification testing';
      case car:
        return 'Car image for classification testing';
      case person:
        return 'Person image for classification testing';
      case street:
        return 'Street scene with multiple objects';
      default:
        return 'Test image';
    }
  }
}
