import 'dart:convert';
import 'package:http/http.dart' as http;

/// Represents a model entry from index.json
class ModelIndexEntry {
  final String name;
  final String modelName;
  final String category;
  final String backend;
  final String hash;
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
    required this.hash,
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
      hash: json['hash'] as String,
      size: json['size'] as int,
      sizeMB: (json['sizeMB'] as num).toDouble(),
      inputSize: json['inputSize'] as int?,
      remoteUrl: json['remoteUrl'] as String,
      platforms: (json['platforms'] as List<dynamic>?)
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
        final modelLabel = modelName.replaceAll('yolo', 'YOLO').replaceAll('v', 'v');
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
  final String hash;
  final int size;
  final String remoteUrl;

  const LabelsIndexEntry({
    required this.name,
    required this.category,
    required this.hash,
    required this.size,
    required this.remoteUrl,
  });

  factory LabelsIndexEntry.fromJson(Map<String, dynamic> json) {
    return LabelsIndexEntry(
      name: json['name'] as String,
      category: json['category'] as String,
      hash: json['hash'] as String,
      size: json['size'] as int,
      remoteUrl: json['remoteUrl'] as String,
    );
  }
}

/// Represents the full model index
class ModelIndex {
  final String version;
  final String generated;
  final String baseUrl;
  final List<ModelIndexEntry> models;
  final List<LabelsIndexEntry> labels;
  final Map<String, dynamic> backends;

  const ModelIndex({
    required this.version,
    required this.generated,
    required this.baseUrl,
    required this.models,
    required this.labels,
    required this.backends,
  });

  factory ModelIndex.fromJson(Map<String, dynamic> json) {
    return ModelIndex(
      version: json['version'] as String,
      generated: json['generated'] as String,
      baseUrl: json['baseUrl'] as String,
      models: (json['models'] as List<dynamic>)
          .map((e) => ModelIndexEntry.fromJson(e as Map<String, dynamic>))
          .toList(),
      labels: (json['labels'] as List<dynamic>)
          .map((e) => LabelsIndexEntry.fromJson(e as Map<String, dynamic>))
          .toList(),
      backends: json['backends'] as Map<String, dynamic>,
    );
  }

  /// Get models for a specific category
  List<ModelIndexEntry> getModelsByCategory(String category) {
    return models.where((m) => m.category == category).toList();
  }

  /// Get models that support a specific platform
  List<ModelIndexEntry> getModelsForPlatform(String platform) {
    return models.where((m) => m.platforms.contains(platform)).toList();
  }

  /// Get labels URL for a category
  String? getLabelsUrl(String category) {
    final label = labels.where((l) => l.category == category).firstOrNull;
    return label?.remoteUrl;
  }

  /// Get labels hash for a category (for cache invalidation)
  String? getLabelsHash(String category) {
    final label = labels.where((l) => l.category == category).firstOrNull;
    return label?.hash;
  }
}

/// Service for fetching and caching the model index
class ModelIndexService {
  static const String _indexUrl =
      'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main/index.json';

  static ModelIndex? _cachedIndex;
  static DateTime? _cacheTime;
  static const Duration _cacheDuration = Duration(hours: 1);

  /// Fetches the model index from the remote server
  /// Caches the result for 1 hour
  static Future<ModelIndex> fetchIndex({bool forceRefresh = false}) async {
    // Return cached index if available and not expired
    if (!forceRefresh &&
        _cachedIndex != null &&
        _cacheTime != null &&
        DateTime.now().difference(_cacheTime!) < _cacheDuration) {
      return _cachedIndex!;
    }

    try {
      final response = await http.get(Uri.parse(_indexUrl));
      if (response.statusCode != 200) {
        throw Exception('Failed to fetch model index: ${response.statusCode}');
      }

      final json = jsonDecode(response.body) as Map<String, dynamic>;
      _cachedIndex = ModelIndex.fromJson(json);
      _cacheTime = DateTime.now();
      return _cachedIndex!;
    } catch (e) {
      // If we have a cached version, return it even if expired
      if (_cachedIndex != null) {
        return _cachedIndex!;
      }
      rethrow;
    }
  }

  /// Clears the cached index
  static void clearCache() {
    _cachedIndex = null;
    _cacheTime = null;
  }
}
