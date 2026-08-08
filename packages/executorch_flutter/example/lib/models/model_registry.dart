import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:universal_platform/universal_platform.dart';

import 'model_definition.dart';
import 'yolo_model_definition.dart';
import 'mobilenet_model_definition.dart';
import 'movenet_model_definition.dart';
import 'blazeface_model_definition.dart';
import 'yolo_pose_model_definition.dart';
import 'yolo_face_model_definition.dart';
import '../services/model_index_service.dart';

/// Central registry of all available models
/// Models are dynamically loaded from index.json hosted on GitHub.
/// Each model is completely self-contained and knows:
/// - Where to download its model file from (GitHub)
/// - Where its labels are (if applicable)
/// - How to process its inputs/outputs
/// - How to render its results
///
/// Backend Information:
/// - XNNPACK: CPU-optimized, works on ALL platforms (Android, iOS, macOS, Web)
/// - CoreML: Apple Neural Engine optimization (iOS, macOS)
/// - Metal: Apple GPU backend (macOS; replaces the removed MPS backend)
/// - Vulkan: Cross-platform GPU (Android, iOS, macOS, Windows, Linux)
///
/// Model Hosting:
/// Models are stored in a separate repository to keep the main repo lightweight.
/// They are downloaded from raw.githubusercontent.com on first use and cached locally.
/// Models repo: https://github.com/abdelaziz-mahdy/executorch_flutter_models
class ModelRegistry {
  /// Base URL for GitHub Releases (model .pte files and labels)
  static const String _releaseBaseUrl =
      'https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases/download';

  /// Fallback version for models when index.json is unavailable
  static const String _fallbackVersion = '1.1.0';

  /// URL for ImageNet class labels (for MobileNet) - stored as release asset
  static const String _imagenetLabelsUrl =
      '$_releaseBaseUrl/v$_fallbackVersion/mobilenet-labels.txt';

  /// URL for COCO class labels (for YOLO) - stored as release asset
  static const String _cocoLabelsUrl =
      '$_releaseBaseUrl/v$_fallbackVersion/yolo-labels.txt';

  /// Loads all available models from the index.json
  ///
  /// [version] - The ExecuTorch version to load models for.
  ///             If null, uses [ModelIndexService.selectedVersion].
  static Future<List<ModelDefinition>> loadAll({String? version}) async {
    try {
      final index = await ModelIndexService.fetchIndex(version: version);
      return _buildModelsFromIndex(index);
    } catch (e) {
      // Fallback to hardcoded models if index fetch fails
      // ignore: avoid_print
      print('Warning: Failed to fetch model index, using fallback: $e');
      return _fallbackModels();
    }
  }

  /// Fetches the list of available ExecuTorch versions
  static Future<ModelVersions> fetchAvailableVersions() async {
    return ModelIndexService.fetchVersions();
  }

  /// Gets the currently selected version
  static String get selectedVersion => ModelIndexService.selectedVersion;

  /// Sets the selected version
  static set selectedVersion(String version) {
    ModelIndexService.selectedVersion = version;
  }

  /// Builds model definitions from the fetched index
  static List<ModelDefinition> _buildModelsFromIndex(ModelIndex index) {
    final currentPlatform = _getCurrentPlatform();
    final models = <ModelDefinition>[];

    // Get available backends on this platform
    final availableBackends = _getAvailableBackends();

    // Filter models by current platform
    final platformModels = index.getModelsForPlatform(currentPlatform);

    for (final entry in platformModels) {
      // Check if the backend is available
      final backend = _stringToBackend(entry.backend);
      if (backend != null && !availableBackends.contains(backend)) {
        // Skip models whose backend is not available
        continue;
      }

      final definition = _createModelDefinition(entry, index);
      if (definition != null) {
        models.add(definition);
      }
    }

    return models;
  }

  /// Gets the list of available backends on the current platform
  static Set<Backend> _getAvailableBackends() {
    try {
      return BackendQuery.available.toSet();
    } catch (e) {
      // If backend query fails, assume all backends are available
      // This prevents blocking model loading on errors
      return Backend.values.toSet();
    }
  }

  /// Converts a backend string to Backend enum
  static Backend? _stringToBackend(String backend) {
    return switch (backend.toLowerCase()) {
      'xnnpack' => Backend.xnnpack,
      'coreml' => Backend.coreml,
      'metal' => Backend.metal,
      // Legacy index entries still say "mps"; Metal replaced it upstream.
      'mps' => Backend.metal,
      'vulkan' => Backend.vulkan,
      'qnn' => Backend.qnn,
      _ => null,
    };
  }

  /// Creates a model definition from an index entry
  static ModelDefinition? _createModelDefinition(
    ModelIndexEntry entry,
    ModelIndex index,
  ) {
    switch (entry.category) {
      case 'mobilenet':
        return MobileNetModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 224,
          hash: entry.hash,
          fileSizeMB: entry.sizeMB,
          labelsRemoteUrl:
              entry.labelsRemoteUrl ??
              index.getLabelsUrl('mobilenet') ??
              _imagenetLabelsUrl,
        );

      case 'yolo':
        return YoloModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 640,
          hash: entry.hash,
          fileSizeMB: entry.sizeMB,
          labelsRemoteUrl:
              entry.labelsRemoteUrl ??
              index.getLabelsUrl('yolo') ??
              _cocoLabelsUrl,
        );

      case 'movenet':
        return MoveNetModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 192,
          hash: entry.hash,
          fileSizeMB: entry.sizeMB,
        );

      case 'blazeface':
        return BlazeFaceModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 128,
          hash: entry.hash,
          fileSizeMB: entry.sizeMB,
        );

      case 'yolo-pose':
        return YoloPoseModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 640,
          hash: entry.hash,
          fileSizeMB: entry.sizeMB,
        );

      case 'yolo-face':
        return YoloFaceModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 640,
          hash: entry.hash,
          fileSizeMB: entry.sizeMB,
        );

      default:
        return null;
    }
  }

  /// Gets the current platform string for filtering
  static String _getCurrentPlatform() {
    if (UniversalPlatform.isWeb) return 'web';
    if (UniversalPlatform.isAndroid) return 'android';
    if (UniversalPlatform.isIOS) return 'ios';
    if (UniversalPlatform.isMacOS) return 'macos';
    if (UniversalPlatform.isLinux) return 'linux';
    if (UniversalPlatform.isWindows) return 'windows';
    return 'android'; // Default fallback
  }

  /// Fallback models if index.json cannot be fetched
  static List<ModelDefinition> _fallbackModels() {
    if (UniversalPlatform.isWeb) {
      return _webFallbackModels();
    }
    return _nativeFallbackModels();
  }

  /// Fallback web models
  static List<ModelDefinition> _webFallbackModels() {
    return [
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_xnnpack',
        displayName: 'MobileNet V3 Small (Web)',
        description:
            'Web-optimized image classification - XNNPACK with WASM SIMD',
        remoteUrl:
            '$_releaseBaseUrl/v$_fallbackVersion/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        fileSizeMB: 9.73,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),
      YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (Web)',
        description: 'Web-optimized object detection - XNNPACK with WASM SIMD',
        remoteUrl: '$_releaseBaseUrl/v$_fallbackVersion/yolo11n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 10.19,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),
    ];
  }

  /// Fallback native models
  static List<ModelDefinition> _nativeFallbackModels() {
    return [
      // MobileNet XNNPACK
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_xnnpack',
        displayName: 'MobileNet V3 Small (XNNPACK)',
        description:
            'CPU-optimized image classification - works on all platforms',
        remoteUrl:
            '$_releaseBaseUrl/v$_fallbackVersion/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        fileSizeMB: 9.73,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),
      // YOLO11n XNNPACK
      YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        remoteUrl: '$_releaseBaseUrl/v$_fallbackVersion/yolo11n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 10.19,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),
    ];
  }
}
