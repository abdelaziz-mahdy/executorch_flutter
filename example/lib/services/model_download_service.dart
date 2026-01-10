import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:universal_platform/universal_platform.dart';

import 'model_download_service_stub.dart'
    if (dart.library.io) 'model_download_service_native.dart'
    if (dart.library.js_interop) 'model_download_service_web.dart'
    as impl;
import 'model_index_service.dart';

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
  final String? localPath; // For native platforms
  final Uint8List? bytes; // For web platform

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

/// Service for downloading and caching models from remote URLs
class ModelDownloadService extends ChangeNotifier {
  ModelDownloadService._();

  static final ModelDownloadService _instance = ModelDownloadService._();
  static ModelDownloadService get instance => _instance;

  final Map<String, ModelDownloadInfo> _downloadStates = {};
  final Map<String, Uint8List> _memoryCache = {}; // For web

  /// Get download state for a model
  ModelDownloadInfo getDownloadInfo(String modelName) {
    return _downloadStates[modelName] ??
        ModelDownloadInfo(
          modelName: modelName,
          state: ModelDownloadState.notDownloaded,
        );
  }

  /// Check if a model is downloaded/cached
  Future<bool> isModelCached(String modelName) async {
    // Check memory cache first (for web)
    if (_memoryCache.containsKey(modelName)) {
      return true;
    }

    // Check platform-specific cache
    return impl.isModelCached(modelName);
  }

  /// Get cached model bytes (for loading)
  Future<Uint8List?> getCachedModelBytes(String modelName) async {
    // Check memory cache first
    if (_memoryCache.containsKey(modelName)) {
      return _memoryCache[modelName];
    }

    // Get from platform-specific cache
    return impl.getCachedModelBytes(modelName);
  }

  /// Download a model from remote URL
  Future<ModelDownloadInfo> downloadModel({
    required String modelName,
    required String remoteUrl,
    DownloadProgressCallback? onProgress,
  }) async {
    // Check if already downloaded
    if (await isModelCached(modelName)) {
      final bytes = await getCachedModelBytes(modelName);
      final info = ModelDownloadInfo(
        modelName: modelName,
        state: ModelDownloadState.downloaded,
        progress: 1.0,
        bytes: bytes,
        localPath: UniversalPlatform.isWeb
            ? null
            : await impl.getLocalPath(modelName),
      );
      _downloadStates[modelName] = info;
      notifyListeners();
      return info;
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

      // Download with progress
      final bytes = await impl.downloadModel(
        modelName: modelName,
        remoteUrl: urlWithCacheBuster,
        onProgress: (progress, received, total) {
          _downloadStates[modelName] = _downloadStates[modelName]!.copyWith(
            progress: progress,
          );
          notifyListeners();
          onProgress?.call(progress, received, total);
        },
      );

      // Cache in memory for web
      if (UniversalPlatform.isWeb) {
        _memoryCache[modelName] = bytes;
      }

      // Update state to downloaded
      final info = ModelDownloadInfo(
        modelName: modelName,
        state: ModelDownloadState.downloaded,
        progress: 1.0,
        bytes: bytes,
        localPath: UniversalPlatform.isWeb
            ? null
            : await impl.getLocalPath(modelName),
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

  /// Delete cached model
  Future<void> deleteModel(String modelName) async {
    _memoryCache.remove(modelName);
    await impl.deleteModel(modelName);
    _downloadStates[modelName] = ModelDownloadInfo(
      modelName: modelName,
      state: ModelDownloadState.notDownloaded,
    );
    notifyListeners();
  }

  /// Clear all cached models
  Future<void> clearCache() async {
    _memoryCache.clear();
    await impl.clearCache();
    _downloadStates.clear();
    notifyListeners();
  }

  /// Get total cache size in bytes
  Future<int> getCacheSize() async {
    int size = 0;

    // Memory cache size
    for (final bytes in _memoryCache.values) {
      size += bytes.length;
    }

    // Platform cache size
    size += await impl.getCacheSize();

    return size;
  }
}
