import 'package:executorch_flutter/executorch_flutter.dart';
import 'base_processor.dart';
import 'image_processor.dart';

/// MobileNet/ImageNet output processor with settings baked in
class MobileNetOutputProcessor extends OutputProcessor<ClassificationResult> {
  const MobileNetOutputProcessor({
    required this.classLabels,
    this.topK = 5,
  });

  final List<String> classLabels;
  final int topK;

  @override
  Future<ClassificationResult> process(List<TensorData> outputs) async {
    final postprocessor = ImageNetPostprocessor(
      classLabels: classLabels,
      topK: topK,
    );
    return await postprocessor.postprocess(outputs);
  }
}
