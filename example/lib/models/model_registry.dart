import 'model_definition.dart';
import 'yolo_model_definition.dart';
import 'mobilenet_model_definition.dart';
import 'gemma_model_definition.dart';

/// Central registry of all available models
/// To add a new model, just add it to this list!
/// Each model is completely self-contained and knows:
/// - Where its model file is
/// - Where its labels are (if applicable)
/// - How to process its inputs/outputs
/// - How to render its results
///
/// Backend Information:
/// - XNNPACK: CPU-optimized, works on all platforms (Android, iOS, macOS)
/// - CoreML: Apple Neural Engine optimization (iOS, macOS)
/// - MPS: Metal Performance Shaders for GPU acceleration (iOS, macOS)
/// - Vulkan: Cross-platform GPU acceleration (Android, Linux)
class ModelRegistry {
  static Future<List<ModelDefinition>> loadAll() async {
    return [
      // ========== MobileNet Models ==========
      // MobileNet V3 Small - XNNPACK (CPU)
      const MobileNetModelDefinition(
        name: 'mobilenet_v3_small_xnnpack',
        displayName: 'MobileNet V3 Small (XNNPACK)',
        description: 'CPU-optimized image classification - works on all platforms',
        assetPath: 'assets/models/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        labelsAssetPath: 'assets/imagenet_classes.txt',
      ),

      // MobileNet V3 Small - CoreML (Apple NPU)
      const MobileNetModelDefinition(
        name: 'mobilenet_v3_small_coreml',
        displayName: 'MobileNet V3 Small (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        assetPath: 'assets/models/mobilenet_v3_small_coreml.pte',
        inputSize: 224,
        labelsAssetPath: 'assets/imagenet_classes.txt',
      ),

      // MobileNet V3 Small - MPS (Apple GPU)
      const MobileNetModelDefinition(
        name: 'mobilenet_v3_small_mps',
        displayName: 'MobileNet V3 Small (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        assetPath: 'assets/models/mobilenet_v3_small_mps.pte',
        inputSize: 224,
        labelsAssetPath: 'assets/imagenet_classes.txt',
      ),

      // MobileNet V3 Small - Vulkan (GPU)
      const MobileNetModelDefinition(
        name: 'mobilenet_v3_small_vulkan',
        displayName: 'MobileNet V3 Small (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        assetPath: 'assets/models/mobilenet_v3_small_vulkan.pte',
        inputSize: 224,
        labelsAssetPath: 'assets/imagenet_classes.txt',
      ),

      // ========== YOLO11 Nano Models ==========
      // YOLO11n - XNNPACK (CPU)
      const YoloModelDefinition(
        name: 'yolo11n_xnnpack',
        displayName: 'YOLO11 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        assetPath: 'assets/models/yolo11n_xnnpack.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLO11n - CoreML (Apple NPU)
      const YoloModelDefinition(
        name: 'yolo11n_coreml',
        displayName: 'YOLO11 Nano (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        assetPath: 'assets/models/yolo11n_coreml.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLO11n - MPS (Apple GPU)
      const YoloModelDefinition(
        name: 'yolo11n_mps',
        displayName: 'YOLO11 Nano (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        assetPath: 'assets/models/yolo11n_mps.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLO11n - Vulkan (GPU)
      const YoloModelDefinition(
        name: 'yolo11n_vulkan',
        displayName: 'YOLO11 Nano (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        assetPath: 'assets/models/yolo11n_vulkan.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // ========== YOLOv8 Nano Models ==========
      // YOLOv8n - XNNPACK (CPU)
      const YoloModelDefinition(
        name: 'yolov8n_xnnpack',
        displayName: 'YOLOv8 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        assetPath: 'assets/models/yolov8n_xnnpack.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLOv8n - CoreML (Apple NPU)
      const YoloModelDefinition(
        name: 'yolov8n_coreml',
        displayName: 'YOLOv8 Nano (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        assetPath: 'assets/models/yolov8n_coreml.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLOv8n - MPS (Apple GPU)
      const YoloModelDefinition(
        name: 'yolov8n_mps',
        displayName: 'YOLOv8 Nano (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        assetPath: 'assets/models/yolov8n_mps.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLOv8n - Vulkan (GPU)
      const YoloModelDefinition(
        name: 'yolov8n_vulkan',
        displayName: 'YOLOv8 Nano (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        assetPath: 'assets/models/yolov8n_vulkan.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // ========== YOLOv5 Nano Models ==========
      // YOLOv5n - XNNPACK (CPU)
      const YoloModelDefinition(
        name: 'yolov5n_xnnpack',
        displayName: 'YOLOv5 Nano (XNNPACK)',
        description: 'CPU-optimized object detection - works on all platforms',
        assetPath: 'assets/models/yolov5n_xnnpack.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLOv5n - CoreML (Apple NPU)
      const YoloModelDefinition(
        name: 'yolov5n_coreml',
        displayName: 'YOLOv5 Nano (CoreML)',
        description: 'Apple Neural Engine optimization - iOS/macOS only',
        assetPath: 'assets/models/yolov5n_coreml.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLOv5n - MPS (Apple GPU)
      const YoloModelDefinition(
        name: 'yolov5n_mps',
        displayName: 'YOLOv5 Nano (MPS)',
        description: 'Metal GPU acceleration - iOS/macOS only',
        assetPath: 'assets/models/yolov5n_mps.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // YOLOv5n - Vulkan (GPU)
      const YoloModelDefinition(
        name: 'yolov5n_vulkan',
        displayName: 'YOLOv5 Nano (Vulkan)',
        description: 'GPU acceleration - Android/Linux',
        assetPath: 'assets/models/yolov5n_vulkan.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // ========== Text Generation Models ==========
      // Note: Text generation models currently only support XNNPACK backend
      const GemmaModelDefinition(
        name: 'gemma-3-270m',
        displayName: 'Gemma 3 270M (Not Working Yet)',
        description: 'Google Gemma 3 text generation model (270M parameters)',
        assetPath: 'assets/models/gemma-3-270m_xnnpack.pte',
        inputSize: 128, // Sequence length
        vocabAssetPath: 'assets/models/gemma-3-270m_vocab.json',
      ),
    ];
  }
}
