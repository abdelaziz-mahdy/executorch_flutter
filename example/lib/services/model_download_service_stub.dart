import 'dart:typed_data';

import 'model_download_service.dart';

/// Stub implementation for unsupported platforms
Future<String?> getLocalPath(String modelName) async {
  throw UnsupportedError('Model download not supported on this platform');
}

Future<bool> isModelCached(String modelName) async {
  return false;
}

Future<Uint8List?> getCachedModelBytes(String modelName) async {
  return null;
}

Future<Uint8List> downloadModel({
  required String modelName,
  required String remoteUrl,
  DownloadProgressCallback? onProgress,
}) async {
  throw UnsupportedError('Model download not supported on this platform');
}

Future<void> deleteModel(String modelName) async {}

Future<void> clearCache() async {}

Future<int> getCacheSize() async {
  return 0;
}
