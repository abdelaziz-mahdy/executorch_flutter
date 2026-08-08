import 'dart:typed_data';

import 'model_download_service.dart';

/// Stub implementation for unsupported platforms

Future<String?> getLocalPath(String path) async {
  throw UnsupportedError('Model cache not supported on this platform');
}

Future<bool> hasModel(String path) async {
  return false;
}

Future<Uint8List?> getModel(String path) async {
  return null;
}

Future<void> saveModel(String path, Uint8List data) async {
  throw UnsupportedError('Model cache not supported on this platform');
}

Future<Uint8List> downloadFromUrl({
  required String remoteUrl,
  DownloadProgressCallback? onProgress,
}) async {
  throw UnsupportedError('Model download not supported on this platform');
}

Future<void> deleteModel(String path) async {}

Future<void> clearAll() async {}

Future<void> clearVersion(String version) async {}

Future<int> getTotalSize() async {
  return 0;
}
