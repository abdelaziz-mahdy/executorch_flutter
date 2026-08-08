import 'dart:convert';
import 'package:executorch_flutter/executorch_flutter.dart'
    show executorchVersion;
import 'package:http/http.dart' as http;

// =============================================================================
// Data Models
// =============================================================================

/// Represents a model entry from index.json
class ModelIndexEntry {
  final String name;
  final String modelName;
  final String category;
  final String backend;
  final String? hash;
  final int size;
  final double sizeMB;
  final int? inputSize;
  final String remoteUrl;
  final List<String> platforms;
  final String? backendDescription;
  final String? labelsFile;
  final String? labelsRemoteUrl;

  const ModelIndexEntry({
    required this.name,
    required this.modelName,
    required this.category,
    required this.backend,
    this.hash,
    required this.size,
    required this.sizeMB,
    required this.inputSize,
    required this.remoteUrl,
    required this.platforms,
    this.backendDescription,
    this.labelsFile,
    this.labelsRemoteUrl,
  });

  factory ModelIndexEntry.fromJson(Map<String, dynamic> json) {
    return ModelIndexEntry(
      name: json['name'] as String,
      modelName: json['modelName'] as String,
      category: json['category'] as String,
      backend: json['backend'] as String,
      hash: json['hash'] as String?,
      size: json['size'] as int,
      sizeMB: (json['sizeMB'] as num).toDouble(),
      inputSize: json['inputSize'] as int?,
      remoteUrl: json['remoteUrl'] as String,
      platforms:
          (json['platforms'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          [],
      backendDescription: json['backendDescription'] as String?,
      labelsFile: json['labelsFile'] as String?,
      labelsRemoteUrl: json['labelsRemoteUrl'] as String?,
    );
  }

  /// Returns a display name for the model
  String get displayName {
    final backendLabel = backend.toUpperCase();
    switch (category) {
      case 'mobilenet':
        return 'MobileNet V3 Small ($backendLabel)';
      case 'yolo':
        final modelLabel = modelName
            .replaceAll('yolo', 'YOLO')
            .replaceAll('v', 'v');
        return '$modelLabel ($backendLabel)';
      case 'gemma':
        return 'Gemma 3 270M ($backendLabel)';
      default:
        return '$modelName ($backendLabel)';
    }
  }

  /// Returns a description for the model
  String get description {
    return backendDescription ?? 'ExecuTorch model with $backend backend';
  }
}

/// Represents a labels entry from index.json
class LabelsIndexEntry {
  final String name;
  final String category;
  final String? hash;
  final int size;
  final String remoteUrl;

  const LabelsIndexEntry({
    required this.name,
    required this.category,
    this.hash,
    required this.size,
    required this.remoteUrl,
  });

  factory LabelsIndexEntry.fromJson(Map<String, dynamic> json) {
    return LabelsIndexEntry(
      name: json['name'] as String,
      category: json['category'] as String,
      hash: json['hash'] as String?,
      size: json['size'] as int,
      remoteUrl: json['remoteUrl'] as String,
    );
  }
}

/// Represents the full model index
class ModelIndex {
  final String version;
  final String? generated;
  final String baseUrl;
  final List<ModelIndexEntry> models;
  final List<LabelsIndexEntry> labels;
  final Map<String, dynamic> backends;

  const ModelIndex({
    required this.version,
    this.generated,
    required this.baseUrl,
    required this.models,
    required this.labels,
    this.backends = const {},
  });

  factory ModelIndex.fromJson(Map<String, dynamic> json) {
    return ModelIndex(
      version: json['version'] as String,
      generated: json['generated'] as String?,
      baseUrl: json['baseUrl'] as String,
      models: (json['models'] as List<dynamic>)
          .map((e) => ModelIndexEntry.fromJson(e as Map<String, dynamic>))
          .toList(),
      labels:
          (json['labels'] as List<dynamic>?)
              ?.map((e) => LabelsIndexEntry.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      backends: json['backends'] as Map<String, dynamic>? ?? {},
    );
  }

  List<ModelIndexEntry> getModelsByCategory(String category) {
    return models.where((m) => m.category == category).toList();
  }

  List<ModelIndexEntry> getModelsForPlatform(String platform) {
    // If platforms list is empty, assume model works on all platforms (backward compatibility)
    return models
        .where((m) => m.platforms.isEmpty || m.platforms.contains(platform))
        .toList();
  }

  String? getLabelsUrl(String category) {
    final label = labels.where((l) => l.category == category).firstOrNull;
    return label?.remoteUrl;
  }

  String? getLabelsHash(String category) {
    final label = labels.where((l) => l.category == category).firstOrNull;
    return label?.hash;
  }
}

/// Represents available model versions
class ModelVersions {
  final List<String> versions;
  final String latest;

  const ModelVersions({required this.versions, required this.latest});

  factory ModelVersions.fromJson(Map<String, dynamic> json) {
    return ModelVersions(
      versions: (json['versions'] as List<dynamic>)
          .map((e) => e as String)
          .toList(),
      latest: json['latest'] as String,
    );
  }
}

// =============================================================================
// Abstract Interface
// =============================================================================

/// Abstract interface for model index data source.
/// Enables decorator pattern for adding cross-cutting concerns.
abstract class ModelIndexDataSource {
  Future<ModelVersions> fetchVersions();
  Future<ModelIndex> fetchIndex(String version);
}

// =============================================================================
// HTTP Client Implementation
// =============================================================================

/// HTTP client implementation of ModelIndexDataSource.
/// Single responsibility: HTTP requests only.
class HttpModelIndexDataSource implements ModelIndexDataSource {
  HttpModelIndexDataSource({String? baseUrl})
    : _baseUrl =
          baseUrl ??
          'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main';

  final String _baseUrl;

  /// Adds a cache-busting timestamp to a URL.
  static String addCacheBuster(String url) {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final separator = url.contains('?') ? '&' : '?';
    return '$url${separator}t=$timestamp';
  }

  @override
  Future<ModelVersions> fetchVersions() async {
    final url = addCacheBuster('$_baseUrl/versions.json');
    final response = await http.get(Uri.parse(url));
    if (response.statusCode != 200) {
      throw Exception('Failed to fetch versions: ${response.statusCode}');
    }
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return ModelVersions.fromJson(json);
  }

  @override
  Future<ModelIndex> fetchIndex(String version) async {
    final url = addCacheBuster('$_baseUrl/$version/index.json');
    final response = await http.get(Uri.parse(url));
    if (response.statusCode != 200) {
      throw Exception('Failed to fetch model index: ${response.statusCode}');
    }
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return ModelIndex.fromJson(json);
  }
}

// =============================================================================
// Caching Decorator
// =============================================================================

/// Cache entry with expiration tracking.
class _CacheEntry<T> {
  _CacheEntry(this.value) : timestamp = DateTime.now();

  final T value;
  final DateTime timestamp;

  bool isExpired(Duration maxAge) =>
      DateTime.now().difference(timestamp) > maxAge;
}

/// Caching decorator for ModelIndexDataSource.
/// Adds time-based caching with stale fallback.
class CachedModelIndexDataSource implements ModelIndexDataSource {
  CachedModelIndexDataSource(
    this._inner, {
    this.cacheDuration = const Duration(hours: 1),
  });

  final ModelIndexDataSource _inner;
  final Duration cacheDuration;

  final Map<String, _CacheEntry<ModelIndex>> _indexCache = {};
  _CacheEntry<ModelVersions>? _versionsCache;

  @override
  Future<ModelVersions> fetchVersions() async {
    // Return valid cache
    final cached = _versionsCache;
    if (cached != null && !cached.isExpired(cacheDuration)) {
      return cached.value;
    }

    try {
      final versions = await _inner.fetchVersions();
      _versionsCache = _CacheEntry(versions);
      return versions;
    } catch (e) {
      // Return stale cache as fallback
      if (cached != null) return cached.value;
      rethrow;
    }
  }

  @override
  Future<ModelIndex> fetchIndex(String version) async {
    // Return valid cache
    final cached = _indexCache[version];
    if (cached != null && !cached.isExpired(cacheDuration)) {
      return cached.value;
    }

    try {
      final index = await _inner.fetchIndex(version);
      _indexCache[version] = _CacheEntry(index);
      return index;
    } catch (e) {
      // Return stale cache as fallback
      if (cached != null) return cached.value;
      rethrow;
    }
  }

  void clearCache() {
    _indexCache.clear();
    _versionsCache = null;
  }

  void clearCacheForVersion(String version) {
    _indexCache.remove(version);
  }
}

// =============================================================================
// Service Facade
// =============================================================================

/// Public service for fetching model index data.
/// Provides a simple static API backed by decorated data source.
class ModelIndexService {
  ModelIndexService._();

  // Decorated data source: HTTP + Caching
  static final CachedModelIndexDataSource _dataSource =
      CachedModelIndexDataSource(HttpModelIndexDataSource());

  /// Currently selected version
  static String selectedVersion = executorchVersion;

  /// Utility for cache-busting URLs (used by other services).
  static String addCacheBuster(String url) =>
      HttpModelIndexDataSource.addCacheBuster(url);

  /// Fetch available versions.
  static Future<ModelVersions> fetchVersions({
    bool forceRefresh = false,
  }) async {
    if (forceRefresh) {
      _dataSource.clearCache();
    }
    try {
      return await _dataSource.fetchVersions();
    } catch (e) {
      // Default fallback
      return ModelVersions(
        versions: [executorchVersion],
        latest: executorchVersion,
      );
    }
  }

  /// Fetch model index for a version.
  static Future<ModelIndex> fetchIndex({
    String? version,
    bool forceRefresh = false,
  }) async {
    final targetVersion = version ?? selectedVersion;
    if (forceRefresh) {
      _dataSource.clearCacheForVersion(targetVersion);
    }
    return _dataSource.fetchIndex(targetVersion);
  }

  /// Clear all cached data.
  static void clearCache() => _dataSource.clearCache();

  /// Clear cache for specific version.
  static void clearCacheForVersion(String version) =>
      _dataSource.clearCacheForVersion(version);
}
