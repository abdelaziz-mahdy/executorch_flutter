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

/// Get local path for a model (path includes version subdirectory)
Future<String?> getLocalPath(String path) async {
  final modelsDir = await _getModelsDir();
  return '${modelsDir.path}/$path.pte';
}

/// Check if a model is cached locally
Future<bool> hasModel(String path) async {
  final localPath = await getLocalPath(path);
  if (localPath == null) return false;
  return File(localPath).exists();
}

/// Get cached model bytes
Future<Uint8List?> getModel(String path) async {
  final localPath = await getLocalPath(path);
  if (localPath == null) return null;
  final file = File(localPath);
  if (await file.exists()) {
    return file.readAsBytes();
  }
  return null;
}

/// Save model to cache
Future<void> saveModel(String path, Uint8List data) async {
  final localPath = await getLocalPath(path);
  if (localPath == null) return;

  final file = File(localPath);

  // Create parent directories if needed (for version subdirectory)
  final parent = file.parent;
  if (!await parent.exists()) {
    await parent.create(recursive: true);
  }

  await file.writeAsBytes(data);
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
  final localPath = await getLocalPath(path);
  if (localPath == null) return;
  final file = File(localPath);
  if (await file.exists()) {
    await file.delete();
  }
}

/// Clear all cached models
Future<void> clearAll() async {
  final modelsDir = await _getModelsDir();
  if (await modelsDir.exists()) {
    await modelsDir.delete(recursive: true);
  }
}

/// Clear cached models for a specific version
Future<void> clearVersion(String version) async {
  final modelsDir = await _getModelsDir();
  final versionDir = Directory('${modelsDir.path}/$version');
  if (await versionDir.exists()) {
    await versionDir.delete(recursive: true);
  }
}

/// Get total cache size
Future<int> getTotalSize() async {
  final modelsDir = await _getModelsDir();
  if (!await modelsDir.exists()) {
    return 0;
  }

  int size = 0;
  await for (final entity in modelsDir.list(recursive: true)) {
    if (entity is File) {
      size += await entity.length();
    }
  }
  return size;
}
