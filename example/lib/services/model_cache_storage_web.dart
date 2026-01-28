import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'model_download_service.dart';

// Web uses in-memory storage organized by version/model path
// Structure: { "1.0.1/mobilenet_v3_small": Uint8List, ... }
final Map<String, Uint8List> _webCache = {};

/// Get local path for a model (not used on web)
Future<String?> getLocalPath(String path) async {
  return null;
}

/// Check if a model is cached
Future<bool> hasModel(String path) async {
  return _webCache.containsKey(path);
}

/// Get cached model bytes
Future<Uint8List?> getModel(String path) async {
  return _webCache[path];
}

/// Save model to cache
Future<void> saveModel(String path, Uint8List data) async {
  _webCache[path] = data;
}

/// Download from URL with progress
Future<Uint8List> downloadFromUrl({
  required String remoteUrl,
  DownloadProgressCallback? onProgress,
}) async {
  final client = http.Client();

  try {
    final request = http.Request('GET', Uri.parse(remoteUrl));
    final response = await client.send(request);

    if (response.statusCode != 200) {
      throw Exception('Failed to download model: HTTP ${response.statusCode}');
    }

    final contentLength = response.contentLength ?? 0;
    final bytes = <int>[];
    int received = 0;

    await for (final chunk in response.stream) {
      bytes.addAll(chunk);
      received += chunk.length;

      if (contentLength > 0) {
        final progress = received / contentLength;
        onProgress?.call(progress, received, contentLength);
      }
    }

    return Uint8List.fromList(bytes);
  } finally {
    client.close();
  }
}

/// Delete a cached model
Future<void> deleteModel(String path) async {
  _webCache.remove(path);
}

/// Clear all cached models
Future<void> clearAll() async {
  _webCache.clear();
}

/// Clear cached models for a specific version
Future<void> clearVersion(String version) async {
  final keysToRemove = _webCache.keys
      .where((key) => key.startsWith('$version/'))
      .toList();
  for (final key in keysToRemove) {
    _webCache.remove(key);
  }
}

/// Get total cache size
Future<int> getTotalSize() async {
  int size = 0;
  for (final bytes in _webCache.values) {
    size += bytes.length;
  }
  return size;
}
