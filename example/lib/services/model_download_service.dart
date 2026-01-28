import 'dart:async';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';

import 'model_cache_storage_stub.dart'
    if (dart.library.io) 'model_cache_storage_native.dart'
    if (dart.library.js_interop) 'model_cache_storage_web.dart'
    as storage;
import 'model_index_service.dart';

// =============================================================================
// Types
// =============================================================================

/// Download progress callback
typedef DownloadProgressCallback =
    void Function(double progress, int received, int total);

/// Model download state
enum ModelDownloadState { notDownloaded, downloading, downloaded, error }

/// Model download info
class ModelDownloadInfo {
  final String modelName;
  final ModelDownloadState state;
  final double progress;
  final String? errorMessage;
  final String? localPath;
  final Uint8List? bytes;

  const ModelDownloadInfo({
    required this.modelName,
    required this.state,
    this.progress = 0.0,
    this.errorMessage,
    this.localPath,
    this.bytes,
  });

  ModelDownloadInfo copyWith({
    String? modelName,
    ModelDownloadState? state,
    double? progress,
    String? errorMessage,
    String? localPath,
    Uint8List? bytes,
  }) {
    return ModelDownloadInfo(
      modelName: modelName ?? this.modelName,
      state: state ?? this.state,
      progress: progress ?? this.progress,
      errorMessage: errorMessage ?? this.errorMessage,
      localPath: localPath ?? this.localPath,
      bytes: bytes ?? this.bytes,
    );
  }

  bool get isReady =>
      state == ModelDownloadState.downloaded &&
      (localPath != null || bytes != null);
}

/// Cache key combining version and model name
class ModelCacheKey {
  final String version;
  final String modelName;

  const ModelCacheKey({required this.version, required this.modelName});

  String get path => '$version/$modelName';

  @override
  bool operator ==(Object other) =>
      other is ModelCacheKey &&
      other.version == version &&
      other.modelName == modelName;

  @override
  int get hashCode => Object.hash(version, modelName);

  @override
  String toString() => 'ModelCacheKey($path)';
}

// =============================================================================
// Abstract Interface
// =============================================================================

/// Abstract interface for model data source.
/// Enables decorator pattern for adding cross-cutting concerns.
abstract class ModelDataSource {
  /// Fetch model bytes from source
  Future<Uint8List> fetchModel(
    ModelCacheKey key,
    String remoteUrl, {
    DownloadProgressCallback? onProgress,
  });

  /// Check if model exists in source
  Future<bool> hasModel(ModelCacheKey key);

  /// Get model bytes if available
  Future<Uint8List?> getModel(ModelCacheKey key);

  /// Delete a specific model
  Future<void> deleteModel(ModelCacheKey key);

  /// Clear all models
  Future<void> clearAll();

  /// Get total size in bytes
  Future<int> getTotalSize();

  /// Get local file path (native only, returns null on web)
  Future<String?> getLocalPath(ModelCacheKey key);
}

// =============================================================================
// HTTP Implementation
// =============================================================================

/// HTTP client implementation of ModelDataSource.
/// Single responsibility: HTTP downloads only, no caching.
class HttpModelDataSource implements ModelDataSource {
  /// Downloads model from remote URL with progress reporting.
  @override
  Future<Uint8List> fetchModel(
    ModelCacheKey key,
    String remoteUrl, {
    DownloadProgressCallback? onProgress,
  }) async {
    return storage.downloadFromUrl(
      remoteUrl: remoteUrl,
      onProgress: onProgress,
    );
  }

  // HTTP source doesn't store models - always returns false/null
  @override
  Future<bool> hasModel(ModelCacheKey key) async => false;

  @override
  Future<Uint8List?> getModel(ModelCacheKey key) async => null;

  @override
  Future<void> deleteModel(ModelCacheKey key) async {}

  @override
  Future<void> clearAll() async {}

  @override
  Future<int> getTotalSize() async => 0;

  @override
  Future<String?> getLocalPath(ModelCacheKey key) async => null;
}

// =============================================================================
// Caching Decorator (Version + Hash Aware)
// =============================================================================

/// Caching decorator for ModelDataSource.
/// Adds version-aware caching with hash verification.
///
/// Cache structure: models/{version}/{modelName}.pte
/// Hash verification ensures cached models match expected content.
class CachedModelDataSource implements ModelDataSource {
  CachedModelDataSource(this._inner);

  final ModelDataSource _inner;

  /// Compute SHA-256 hash of data
  static String computeHash(Uint8List data) {
    return sha256.convert(data).toString();
  }

  /// Verify data matches expected hash
  static bool verifyHash(Uint8List data, String expectedHash) {
    final actualHash = computeHash(data);
    return actualHash == expectedHash;
  }

  @override
  Future<Uint8List> fetchModel(
    ModelCacheKey key,
    String remoteUrl, {
    DownloadProgressCallback? onProgress,
    String? expectedHash,
  }) async {
    // Check cache first
    final cached = await getModel(key);
    if (cached != null) {
      // Verify hash if provided
      if (expectedHash != null && !verifyHash(cached, expectedHash)) {
        // Hash mismatch - delete stale cache and re-download
        await deleteModel(key);
      } else {
        return cached;
      }
    }

    // Download from inner source
    final data = await _inner.fetchModel(key, remoteUrl, onProgress: onProgress);

    // Verify downloaded data if hash provided
    if (expectedHash != null && !verifyHash(data, expectedHash)) {
      throw Exception(
        'Downloaded model hash mismatch. Expected: $expectedHash, '
        'Got: ${computeHash(data)}',
      );
    }

    // Save to cache
    await storage.saveModel(key.path, data);

    return data;
  }

  /// Fetch model with hash verification
  Future<Uint8List> fetchModelWithHash(
    ModelCacheKey key,
    String remoteUrl,
    String expectedHash, {
    DownloadProgressCallback? onProgress,
  }) async {
    return fetchModel(
      key,
      remoteUrl,
      onProgress: onProgress,
      expectedHash: expectedHash,
    );
  }

  @override
  Future<bool> hasModel(ModelCacheKey key) async {
    return storage.hasModel(key.path);
  }

  @override
  Future<Uint8List?> getModel(ModelCacheKey key) async {
    return storage.getModel(key.path);
  }

  @override
  Future<void> deleteModel(ModelCacheKey key) async {
    await storage.deleteModel(key.path);
  }

  @override
  Future<void> clearAll() async {
    await storage.clearAll();
  }

  @override
  Future<int> getTotalSize() async {
    return storage.getTotalSize();
  }

  @override
  Future<String?> getLocalPath(ModelCacheKey key) async {
    return storage.getLocalPath(key.path);
  }

  /// Clear cache for a specific version only
  Future<void> clearVersion(String version) async {
    await storage.clearVersion(version);
  }
}

// =============================================================================
// Service Facade
// =============================================================================

/// Public service for downloading and caching models.
/// Provides stateful API with download progress and ChangeNotifier.
class ModelDownloadService extends ChangeNotifier {
  ModelDownloadService._();

  static final ModelDownloadService _instance = ModelDownloadService._();
  static ModelDownloadService get instance => _instance;

  // Decorated data source: HTTP + Caching
  static final CachedModelDataSource _dataSource =
      CachedModelDataSource(HttpModelDataSource());

  final Map<String, ModelDownloadInfo> _downloadStates = {};

  /// Get download state for a model
  ModelDownloadInfo getDownloadInfo(String modelName) {
    return _downloadStates[modelName] ??
        ModelDownloadInfo(
          modelName: modelName,
          state: ModelDownloadState.notDownloaded,
        );
  }

  /// Check if a model is downloaded/cached for a specific version
  Future<bool> isModelCached(String modelName, {String? version}) async {
    final targetVersion = version ?? ModelIndexService.selectedVersion;
    final key = ModelCacheKey(version: targetVersion, modelName: modelName);
    return _dataSource.hasModel(key);
  }

  /// Get cached model bytes
  Future<Uint8List?> getCachedModelBytes(
    String modelName, {
    String? version,
  }) async {
    final targetVersion = version ?? ModelIndexService.selectedVersion;
    final key = ModelCacheKey(version: targetVersion, modelName: modelName);
    return _dataSource.getModel(key);
  }

  /// Download a model from remote URL with version-aware caching and hash verification
  Future<ModelDownloadInfo> downloadModel({
    required String modelName,
    required String remoteUrl,
    String? version,
    String? expectedHash,
    DownloadProgressCallback? onProgress,
  }) async {
    final targetVersion = version ?? ModelIndexService.selectedVersion;
    final key = ModelCacheKey(version: targetVersion, modelName: modelName);

    // Check if already cached (with valid hash if provided)
    if (await _dataSource.hasModel(key)) {
      final cached = await _dataSource.getModel(key);
      if (cached != null) {
        // Verify hash if provided
        if (expectedHash == null ||
            CachedModelDataSource.verifyHash(cached, expectedHash)) {
          final info = ModelDownloadInfo(
            modelName: modelName,
            state: ModelDownloadState.downloaded,
            progress: 1.0,
            bytes: cached,
            localPath: await _dataSource.getLocalPath(key),
          );
          _downloadStates[modelName] = info;
          notifyListeners();
          return info;
        } else {
          // Hash mismatch - will re-download
          await _dataSource.deleteModel(key);
        }
      }
    }

    // Update state to downloading
    _downloadStates[modelName] = ModelDownloadInfo(
      modelName: modelName,
      state: ModelDownloadState.downloading,
      progress: 0.0,
    );
    notifyListeners();

    try {
      // Add cache buster to invalidate CDN cache
      final urlWithCacheBuster = ModelIndexService.addCacheBuster(remoteUrl);

      // Download with caching and hash verification
      final bytes = await _dataSource.fetchModelWithHash(
        key,
        urlWithCacheBuster,
        expectedHash ?? '', // Empty string skips verification
        onProgress: (progress, received, total) {
          _downloadStates[modelName] = _downloadStates[modelName]!.copyWith(
            progress: progress,
          );
          notifyListeners();
          onProgress?.call(progress, received, total);
        },
      );

      // Update state to downloaded
      final info = ModelDownloadInfo(
        modelName: modelName,
        state: ModelDownloadState.downloaded,
        progress: 1.0,
        bytes: bytes,
        localPath: await _dataSource.getLocalPath(key),
      );
      _downloadStates[modelName] = info;
      notifyListeners();
      return info;
    } catch (e) {
      final info = ModelDownloadInfo(
        modelName: modelName,
        state: ModelDownloadState.error,
        errorMessage: e.toString(),
      );
      _downloadStates[modelName] = info;
      notifyListeners();
      return info;
    }
  }

  /// Delete cached model for a specific version
  Future<void> deleteModel(String modelName, {String? version}) async {
    final targetVersion = version ?? ModelIndexService.selectedVersion;
    final key = ModelCacheKey(version: targetVersion, modelName: modelName);
    await _dataSource.deleteModel(key);
    _downloadStates[modelName] = ModelDownloadInfo(
      modelName: modelName,
      state: ModelDownloadState.notDownloaded,
    );
    notifyListeners();
  }

  /// Clear all cached models
  Future<void> clearCache() async {
    await _dataSource.clearAll();
    _downloadStates.clear();
    notifyListeners();
  }

  /// Clear cached models for a specific version
  Future<void> clearCacheForVersion(String version) async {
    await _dataSource.clearVersion(version);
    _downloadStates.clear();
    notifyListeners();
  }

  /// Get total cache size in bytes
  Future<int> getCacheSize() async {
    return _dataSource.getTotalSize();
  }
}
