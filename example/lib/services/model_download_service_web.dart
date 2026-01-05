import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'model_download_service.dart';

// Web uses in-memory storage only (IndexedDB could be added for persistence)
final Map<String, Uint8List> _webCache = {};

/// Get local path for a model (not used on web)
Future<String?> getLocalPath(String modelName) async {
  return null;
}

/// Check if a model is cached
Future<bool> isModelCached(String modelName) async {
  return _webCache.containsKey(modelName);
}

/// Get cached model bytes
Future<Uint8List?> getCachedModelBytes(String modelName) async {
  return _webCache[modelName];
}

/// Download model from remote URL
Future<Uint8List> downloadModel({
  required String modelName,
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

    final data = Uint8List.fromList(bytes);

    // Cache in memory
    _webCache[modelName] = data;

    return data;
  } finally {
    client.close();
  }
}

/// Delete a cached model
Future<void> deleteModel(String modelName) async {
  _webCache.remove(modelName);
}

/// Clear all cached models
Future<void> clearCache() async {
  _webCache.clear();
}

/// Get total cache size
Future<int> getCacheSize() async {
  int size = 0;
  for (final bytes in _webCache.values) {
    size += bytes.length;
  }
  return size;
}
