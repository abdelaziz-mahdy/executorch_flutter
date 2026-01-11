import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'model_download_service.dart';

/// Get the models cache directory
Future<Directory> _getModelsDir() async {
  final cacheDir = await getApplicationCacheDirectory();
  final modelsDir = Directory('${cacheDir.path}/models');
  if (!await modelsDir.exists()) {
    await modelsDir.create(recursive: true);
  }
  return modelsDir;
}

/// Get local path for a model
Future<String> getLocalPath(String modelName) async {
  final modelsDir = await _getModelsDir();
  return '${modelsDir.path}/$modelName.pte';
}

/// Check if a model is cached locally
Future<bool> isModelCached(String modelName) async {
  final path = await getLocalPath(modelName);
  return File(path).exists();
}

/// Get cached model bytes
Future<Uint8List?> getCachedModelBytes(String modelName) async {
  final path = await getLocalPath(modelName);
  final file = File(path);
  if (await file.exists()) {
    return file.readAsBytes();
  }
  return null;
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

    // Save to cache
    final path = await getLocalPath(modelName);
    await File(path).writeAsBytes(data);

    return data;
  } finally {
    client.close();
  }
}

/// Delete a cached model
Future<void> deleteModel(String modelName) async {
  final path = await getLocalPath(modelName);
  final file = File(path);
  if (await file.exists()) {
    await file.delete();
  }
}

/// Clear all cached models
Future<void> clearCache() async {
  final modelsDir = await _getModelsDir();
  if (await modelsDir.exists()) {
    await modelsDir.delete(recursive: true);
  }
}

/// Get total cache size
Future<int> getCacheSize() async {
  final modelsDir = await _getModelsDir();
  if (!await modelsDir.exists()) {
    return 0;
  }

  int size = 0;
  await for (final entity in modelsDir.list()) {
    if (entity is File) {
      size += await entity.length();
    }
  }
  return size;
}
