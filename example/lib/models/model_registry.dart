import 'package:universal_platform/universal_platform.dart';

import 'model_definition.dart';
import 'yolo_model_definition.dart';
import 'mobilenet_model_definition.dart';
import 'gemma_model_definition.dart';

/// Central registry of all available models
/// To add a new model, just add it to this list!
/// Each model is completely self-contained and knows:
/// - Where to download its model file from (GitHub)
/// - Where its labels are (if applicable)
/// - How to process its inputs/outputs
/// - How to render its results
///
/// Backend Information:
/// - Portable: Generic CPU backend, works on web/wasm
/// - XNNPACK: CPU-optimized, works on native platforms (Android, iOS, macOS)
/// - CoreML: Apple Neural Engine optimization (iOS, macOS)
/// - MPS: Metal Performance Shaders for GPU acceleration (iOS, macOS)
/// - Vulkan: Cross-platform GPU acceleration (Android, Linux)
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

  /// MobileNet models directory
  static const String _mobilenetDir = '$_baseUrl/mobilenet';

  /// YOLO models directory
  static const String _yoloDir = '$_baseUrl/yolo';

  /// URL for ImageNet class labels (for MobileNet)
  static const String _imagenetLabelsUrl = '$_mobilenetDir/labels.txt';

  /// URL for COCO class labels (for YOLO)
  static const String _cocoLabelsUrl = '$_yoloDir/labels.txt';

  static Future<List<ModelDefinition>> loadAll() async {
    // Web platform uses portable backend models
    if (UniversalPlatform.isWeb) {
      return _webModels();
    }

    // Native platforms use optimized backend models
    return _nativeModels();
  }

  /// Models optimized for web platform (XNNPACK backend with WASM SIMD)
  static List<ModelDefinition> _webModels() {
    return [
      // ========== MobileNet Models (Web) ==========
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_xnnpack',
        displayName: 'MobileNet V3 Small (Web)',
        description: 'Web-optimized image classification - XNNPACK with WASM SIMD',
        remoteUrl: '$_mobilenetDir/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        fileSizeMB: 9.5,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),

      // ========== YOLO11 Nano Models (Web) ==========
      YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (Web)',
        description: 'Web-optimized object detection - XNNPACK with WASM SIMD',
        remoteUrl: '$_yoloDir/yolo11n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 5.4,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // ========== YOLOv8 Nano Models (Web) ==========
      YoloModelDefinition(
        name: 'yolov8n_xnnpack',
        displayName: 'YOLOv8 Nano (Web)',
        description: 'Web-optimized object detection - XNNPACK with WASM SIMD',
        remoteUrl: '$_yoloDir/yolov8n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 6.2,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // ========== YOLOv5 Nano Models (Web) ==========
      YoloModelDefinition(
        name: 'yolov5n_xnnpack',
        displayName: 'YOLOv5 Nano (Web)',
        description: 'Web-optimized object detection - XNNPACK with WASM SIMD',
        remoteUrl: '$_yoloDir/yolov5n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 3.9,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),
    ];
  }

  /// Models optimized for native platforms (XNNPACK, CoreML, MPS, Vulkan)
  static List<ModelDefinition> _nativeModels() {
    return [
      // ========== MobileNet Models ==========
      // MobileNet V3 Small - XNNPACK (CPU)
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_xnnpack',
        displayName: 'MobileNet V3 Small (XNNPACK)',
        description:
            'CPU-optimized image classification - works on all platforms',
        remoteUrl: '$_mobilenetDir/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        fileSizeMB: 9.5,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),

      // MobileNet V3 Small - CoreML (Apple NPU)
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_coreml',
        displayName: 'MobileNet V3 Small (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        remoteUrl: '$_mobilenetDir/mobilenet_v3_small_coreml.pte',
        inputSize: 224,
        fileSizeMB: 10.2,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),

      // MobileNet V3 Small - MPS (Apple GPU)
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_mps',
        displayName: 'MobileNet V3 Small (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        remoteUrl: '$_mobilenetDir/mobilenet_v3_small_mps.pte',
        inputSize: 224,
        fileSizeMB: 9.8,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),

      // MobileNet V3 Small - Vulkan (GPU)
      MobileNetModelDefinition(
        name: 'mobilenet_v3_small_vulkan',
        displayName: 'MobileNet V3 Small (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        remoteUrl: '$_mobilenetDir/mobilenet_v3_small_vulkan.pte',
        inputSize: 224,
        fileSizeMB: 9.6,
        labelsRemoteUrl: _imagenetLabelsUrl,
      ),

      // ========== YOLO11 Nano Models ==========
      // YOLO11n - XNNPACK (CPU)
      YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        remoteUrl: '$_yoloDir/yolo11n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 5.4,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLO11n - CoreML (Apple NPU)
      YoloModelDefinition(
        name: 'yolo11n_coreml',
        displayName: 'YOLO11 Nano (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        remoteUrl: '$_yoloDir/yolo11n_coreml.pte',
        inputSize: 640,
        fileSizeMB: 5.8,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLO11n - MPS (Apple GPU)
      YoloModelDefinition(
        name: 'yolo11n_mps',
        displayName: 'YOLO11 Nano (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        remoteUrl: '$_yoloDir/yolo11n_mps.pte',
        inputSize: 640,
        fileSizeMB: 5.6,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLO11n - Vulkan (GPU)
      YoloModelDefinition(
        name: 'yolo11n_vulkan',
        displayName: 'YOLO11 Nano (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        remoteUrl: '$_yoloDir/yolo11n_vulkan.pte',
        inputSize: 640,
        fileSizeMB: 5.5,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // ========== YOLOv8 Nano Models ==========
      // YOLOv8n - XNNPACK (CPU)
      YoloModelDefinition(
        name: 'yolov8n_xnnpack',
        displayName: 'YOLOv8 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        remoteUrl: '$_yoloDir/yolov8n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 6.2,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLOv8n - CoreML (Apple NPU)
      YoloModelDefinition(
        name: 'yolov8n_coreml',
        displayName: 'YOLOv8 Nano (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        remoteUrl: '$_yoloDir/yolov8n_coreml.pte',
        inputSize: 640,
        fileSizeMB: 6.6,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLOv8n - MPS (Apple GPU)
      YoloModelDefinition(
        name: 'yolov8n_mps',
        displayName: 'YOLOv8 Nano (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        remoteUrl: '$_yoloDir/yolov8n_mps.pte',
        inputSize: 640,
        fileSizeMB: 6.4,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLOv8n - Vulkan (GPU)
      YoloModelDefinition(
        name: 'yolov8n_vulkan',
        displayName: 'YOLOv8 Nano (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        remoteUrl: '$_yoloDir/yolov8n_vulkan.pte',
        inputSize: 640,
        fileSizeMB: 6.3,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // ========== YOLOv5 Nano Models ==========
      // YOLOv5n - XNNPACK (CPU)
      YoloModelDefinition(
        name: 'yolov5n_xnnpack',
        displayName: 'YOLOv5 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        remoteUrl: '$_yoloDir/yolov5n_xnnpack.pte',
        inputSize: 640,
        fileSizeMB: 3.9,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLOv5n - CoreML (Apple NPU)
      YoloModelDefinition(
        name: 'yolov5n_coreml',
        displayName: 'YOLOv5 Nano (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        remoteUrl: '$_yoloDir/yolov5n_coreml.pte',
        inputSize: 640,
        fileSizeMB: 4.2,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLOv5n - MPS (Apple GPU)
      YoloModelDefinition(
        name: 'yolov5n_mps',
        displayName: 'YOLOv5 Nano (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        remoteUrl: '$_yoloDir/yolov5n_mps.pte',
        inputSize: 640,
        fileSizeMB: 4.0,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // YOLOv5n - Vulkan (GPU)
      YoloModelDefinition(
        name: 'yolov5n_vulkan',
        displayName: 'YOLOv5 Nano (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        remoteUrl: '$_yoloDir/yolov5n_vulkan.pte',
        inputSize: 640,
        fileSizeMB: 4.0,
        labelsRemoteUrl: _cocoLabelsUrl,
      ),

      // ========== Text Generation Models ==========
      // Note: Text generation models currently only support XNNPACK backend
      GemmaModelDefinition(
        name: 'gemma-3-270m',
        displayName: 'Gemma 3 270M (Not Working Yet)',
        description: 'Google Gemma 3 text generation model (270M parameters)',
        remoteUrl: '$_baseUrl/gemma/gemma-3-270m_xnnpack.pte',
        inputSize: 128, // Sequence length
        fileSizeMB: 540.0,
        vocabRemoteUrl: '$_baseUrl/gemma/vocab.json',
      ),
    ];
  }
}
