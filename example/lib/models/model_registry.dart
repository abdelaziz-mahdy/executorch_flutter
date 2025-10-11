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
class ModelRegistry {
  static Future<List<ModelDefinition>> loadAll() async {
    return [
      // MobileNet Models
      const MobileNetModelDefinition(
        name: 'mobilenet_v3_small',
        displayName: 'MobileNet V3 Small',
        description: 'Lightweight image classification model',
        assetPath: 'assets/models/mobilenet_v3_small_xnnpack.pte',
        inputSize: 224,
        labelsAssetPath: 'assets/imagenet_classes.txt',
      ),

      // YOLO Models
      const YoloModelDefinition(
        name: 'yolo11n',
        displayName: 'YOLO11 Nano',
        description: 'Latest YOLO object detection model',
        assetPath: 'assets/models/yolo11n_xnnpack.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),
      const YoloModelDefinition(
        name: 'yolov8n',
        displayName: 'YOLOv8 Nano',
        description: 'YOLOv8 object detection model',
        assetPath: 'assets/models/yolov8n_xnnpack.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),
      const YoloModelDefinition(
        name: 'yolov5n',
        displayName: 'YOLOv5 Nano',
        description: 'YOLOv5 object detection model',
        assetPath: 'assets/models/yolov5n_xnnpack.pte',
        inputSize: 640,
        labelsAssetPath: 'assets/coco_labels.txt',
      ),

      // Text Generation Models
      const GemmaModelDefinition(
        name: 'gemma-3-270m',
        displayName: 'Gemma 3 270M',
        description: 'Google Gemma 3 text generation model (270M parameters)',
        assetPath: 'assets/models/gemma-3-270m_xnnpack.pte',
        inputSize: 128, // Sequence length
        vocabAssetPath: 'assets/models/gemma-3-270m_vocab.json',
      ),
    ];
  }
}
