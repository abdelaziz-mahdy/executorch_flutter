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
/// - MPS: Metal Performance Shaders for GPU acceleration (iOS, macOS)
/// - Vulkan: Cross-platform GPU (Android, iOS, macOS, Windows, Linux)
///
/// Model Hosting:
/// Models are stored in a separate repository to keep the main repo lightweight.
/// They are downloaded from raw.githubusercontent.com on first use and cached locally.
/// Models repo: https://github.com/abdelaziz-mahdy/executorch_flutter_models
class ModelRegistry {
  /// Base URL for model downloads from GitHub raw content
  /// Models are hosted in a separate repository for faster cloning of main repo
  static const String _baseUrl =
      'https://raw.githubusercontent.com/abdelaziz-mahdy/executorch_flutter_models/main';

  /// URL for ImageNet class labels (for MobileNet)
  static const String _imagenetLabelsUrl = '$_baseUrl/mobilenet/labels.txt';

  /// URL for COCO class labels (for YOLO)
  static const String _cocoLabelsUrl = '$_baseUrl/yolo/labels.txt';

  /// Loads all available models from the index.json
  static Future<List<ModelDefinition>> loadAll() async {
    try {
      final index = await ModelIndexService.fetchIndex();
      return _buildModelsFromIndex(index);
    } catch (e) {
      // Fallback to hardcoded models if index fetch fails
      // ignore: avoid_print
      print('Warning: Failed to fetch model index, using fallback: $e');
      return _fallbackModels();
    }
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
      'mps' => Backend.mps,
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
          fileSizeMB: entry.sizeMB,
          labelsRemoteUrl: entry.labelsRemoteUrl ??
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
          fileSizeMB: entry.sizeMB,
          labelsRemoteUrl: entry.labelsRemoteUrl ??
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
          fileSizeMB: entry.sizeMB,
        );

      case 'blazeface':
        return BlazeFaceModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 128,
          fileSizeMB: entry.sizeMB,
        );

      case 'yolo-pose':
        return YoloPoseModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 640,
          fileSizeMB: entry.sizeMB,
        );

      case 'yolo-face':
        return YoloFaceModelDefinition(
          name: '${entry.modelName}_${entry.backend}',
          displayName: entry.displayName,
          description: entry.description,
          remoteUrl: entry.remoteUrl,
          inputSize: entry.inputSize ?? 640,
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
        description: 'Web-optimized image classification - XNNPACK with WASM SIMD',
        remoteUrl: '$_baseUrl/mobilenet/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        fileSizeMB: 9.73,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),
      YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (Web)',
        description: 'Web-optimized object detection - XNNPACK with WASM SIMD',
        remoteUrl: '$_baseUrl/yolo/yolo11n_xnnpack.pte',
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
        description: 'CPU-optimized image classification - works on all platforms',
        remoteUrl: '$_baseUrl/mobilenet/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        fileSizeMB: 9.73,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),
      // YOLO11n XNNPACK
      YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        remoteUrl: '$_baseUrl/yolo/yolo11n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 10.19,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),
    ];
  }
}
