import 'package:executorch_flutter/executorch_flutter.dart';
import 'base_processor.dart';
import 'yolo_processor.dart';

/// YOLO output processor with settings baked in
class YoloOutputProcessor extends OutputProcessor<ObjectDetectionResult> {
  const YoloOutputProcessor({
    required this.classLabels,
    required this.inputWidth,
    required this.inputHeight,
    required this.confidenceThreshold,
    required this.iouThreshold,
  });

  final List<String> classLabels;
  final int inputWidth;
  final int inputHeight;
  final double confidenceThreshold;
  final double iouThreshold;

  @override
  Future<ObjectDetectionResult> process(List<TensorData> outputs) async {
    final postprocessor = YoloPostprocessor(
      classLabels: classLabels,
      inputWidth: inputWidth,
      inputHeight: inputHeight,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
    );

    return await postprocessor.postprocess(outputs);
  }
}
