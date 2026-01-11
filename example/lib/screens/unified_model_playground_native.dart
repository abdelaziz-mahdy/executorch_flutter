/// Native platform helpers for unified model playground
library;

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// Load a model asset to a file path (native platforms only)
Future<String> loadAssetToFile(String assetPath) async {
  try {
    final byteData = await rootBundle.load(assetPath);
    final directory = await getApplicationCacheDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${directory.path}/$fileName');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  } catch (e) {
    throw Exception('Failed to load asset "$assetPath": $e');
  }
}
