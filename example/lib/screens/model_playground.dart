import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../processors/processors.dart';
import '../services/performance_service.dart';

/// Supported model types in the playground
enum ModelType {
  imageClassification('Image Classification', Icons.image, 'Classify images using ImageNet'),
  objectDetection('Object Detection', Icons.center_focus_strong, 'Detect and locate objects with YOLO');

  const ModelType(this.displayName, this.icon, this.description);
  final String displayName;
  final IconData icon;
  final String description;
}

/// Model configuration
class ModelConfig {
  const ModelConfig({
    required this.name,
    required this.type,
    required this.assetPath,
    required this.inputSize,
  });

  final String name;
  final ModelType type;
  final String assetPath;
  final int inputSize;
}

/// Available models
const List<ModelConfig> availableModels = [
  ModelConfig(
    name: 'MobileNet V3',
    type: ModelType.imageClassification,
    assetPath: 'assets/models/mobilenet_v3_small_xnnpack.pte',
    inputSize: 224,
  ),
  ModelConfig(
    name: 'YOLO12 Nano',
    type: ModelType.objectDetection,
    assetPath: 'assets/models/yolo12n_xnnpack.pte',
    inputSize: 640,
  ),
  ModelConfig(
    name: 'YOLO11 Nano',
    type: ModelType.objectDetection,
    assetPath: 'assets/models/yolo11n_xnnpack.pte',
    inputSize: 640,
  ),
  ModelConfig(
    name: 'YOLOv8 Nano',
    type: ModelType.objectDetection,
    assetPath: 'assets/models/yolov8n_xnnpack.pte',
    inputSize: 640,
  ),
  ModelConfig(
    name: 'YOLOv5 Nano',
    type: ModelType.objectDetection,
    assetPath: 'assets/models/yolov5n_xnnpack.pte',
    inputSize: 640,
  ),
];

/// Modern single-page model playground
class ModelPlayground extends StatefulWidget {
  const ModelPlayground({super.key});

  @override
  State<ModelPlayground> createState() => _ModelPlaygroundState();
}

class _ModelPlaygroundState extends State<ModelPlayground> with SingleTickerProviderStateMixin {
  // Model state
  ModelConfig? _selectedModelConfig;
  ExecuTorchModel? _loadedModel;
  List<String>? _classLabels;
  List<String>? _cocoLabels;

  // Processing state
  bool _isLoading = false;
  bool _isProcessing = false;
  File? _selectedImage;

  // Results
  ClassificationResult? _classificationResult;
  ObjectDetectionResult? _objectDetectionResult;
  String? _errorMessage;
  double? _processingTime;

  // Animation
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _fadeAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    );
    _loadClassLabels();
  }

  @override
  void dispose() {
    _animationController.dispose();
    _loadedModel?.dispose();
    super.dispose();
  }

  Future<void> _loadClassLabels() async {
    try {
      // Load ImageNet labels (for classification models)
      final imageNetLabels = await rootBundle.loadString('assets/imagenet_classes.txt');

      // Load COCO labels (for YOLO detection models)
      final cocoLabels = await rootBundle.loadString('assets/coco_labels.txt');

      setState(() {
        // Use ImageNet by default, will switch to COCO for object detection
        _classLabels = imageNetLabels.split('\n').where((line) => line.isNotEmpty).toList();
        _cocoLabels = cocoLabels.split('\n').where((line) => line.isNotEmpty).toList();
      });
    } catch (e) {
      debugPrint('Failed to load class labels: $e');
    }
  }

  Future<void> _loadModel(ModelConfig config) async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _classificationResult = null;
      _objectDetectionResult = null;
    });

    try {
      // Dispose previous model
      await _loadedModel?.dispose();

      // Load model asset
      final modelPath = await _loadAssetModel(config.assetPath);

      // Load model
      final model = await ExecutorchManager.instance.loadModel(modelPath);

      setState(() {
        _selectedModelConfig = config;
        _loadedModel = model;
        _isLoading = false;
      });

      _animationController.forward();
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load model: $e';
        _isLoading = false;
      });
    }
  }

  Future<String> _loadAssetModel(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final directory = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${directory.path}/$fileName');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: source);

      if (image != null) {
        final file = File(image.path);
        setState(() {
          _selectedImage = file;
          _classificationResult = null;
          _objectDetectionResult = null;
          _errorMessage = null;
        });
        await _processImage(file);
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to pick image: $e';
      });
    }
  }

  Future<void> _processImage(File imageFile) async {
    if (_loadedModel == null || _selectedModelConfig == null) return;

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    final stopwatch = Stopwatch()..start();

    try {
      final imageBytes = await imageFile.readAsBytes();

      if (_selectedModelConfig!.type == ModelType.imageClassification) {
        await _runClassification(imageBytes);
      } else if (_selectedModelConfig!.type == ModelType.objectDetection) {
        await _runObjectDetection(imageBytes);
      }

      stopwatch.stop();
      setState(() {
        _processingTime = stopwatch.elapsedMilliseconds.toDouble();
      });

      PerformanceService().recordProcessingTime(_processingTime!);
    } catch (e) {
      setState(() {
        _errorMessage = 'Processing failed: $e';
      });
      debugPrint('❌ Processing error: $e');
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Future<void> _runClassification(Uint8List imageBytes) async {
    final processor = ImageNetProcessor(
      preprocessConfig: ImagePreprocessConfig(
        targetWidth: _selectedModelConfig!.inputSize,
        targetHeight: _selectedModelConfig!.inputSize,
        normalizeToFloat: true,
        meanSubtraction: [0.485, 0.456, 0.406],
        standardDeviation: [0.229, 0.224, 0.225],
        cropMode: ImageCropMode.centerCrop,
      ),
      classLabels: _classLabels ?? [],
    );

    final result = await processor.process(imageBytes, _loadedModel!);

    setState(() {
      _classificationResult = result;
      _objectDetectionResult = null;
    });
  }

  Future<void> _runObjectDetection(Uint8List imageBytes) async {
    final processor = YoloProcessor(
      preprocessConfig: YoloPreprocessConfig(
        targetWidth: _selectedModelConfig!.inputSize,
        targetHeight: _selectedModelConfig!.inputSize,
      ),
      classLabels: _cocoLabels ?? _classLabels ?? [],
      confidenceThreshold: 0.25,
      iouThreshold: 0.45,
      maxDetections: 300,
    );

    final result = await processor.process(imageBytes, _loadedModel!);

    setState(() {
      _objectDetectionResult = result;
      _classificationResult = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.surface,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: _selectedModelConfig == null
                  ? _buildModelSelection()
                  : _buildModelWorkspace(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primaryContainer,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.primary,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              Icons.psychology,
              color: Theme.of(context).colorScheme.onPrimary,
              size: 28,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'ExecuTorch Playground',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).colorScheme.onPrimaryContainer,
                  ),
                ),
                Text(
                  _selectedModelConfig?.name ?? 'Select a model to begin',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Theme.of(context).colorScheme.onPrimaryContainer.withOpacity(0.7),
                  ),
                ),
              ],
            ),
          ),
          if (_selectedModelConfig != null)
            IconButton(
              icon: const Icon(Icons.close),
              onPressed: () {
                setState(() {
                  _selectedModelConfig = null;
                  _loadedModel?.dispose();
                  _loadedModel = null;
                  _selectedImage = null;
                  _classificationResult = null;
                  _objectDetectionResult = null;
                  _errorMessage = null;
                });
                _animationController.reverse();
              },
              tooltip: 'Change model',
            ),
        ],
      ),
    );
  }

  Widget _buildModelSelection() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Choose a Model',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Select a machine learning model to start processing images',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
            ),
          ),
          const SizedBox(height: 32),
          ...availableModels.map((model) => Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: _buildModelCard(model),
          )),
        ],
      ),
    );
  }

  Widget _buildModelCard(ModelConfig config) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: InkWell(
        onTap: () => _loadModel(config),
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primaryContainer.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  config.type.icon,
                  size: 32,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      config.name,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      config.type.displayName,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Theme.of(context).colorScheme.primary,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      config.type.description,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(
                Icons.arrow_forward_ios,
                size: 16,
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.3),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildModelWorkspace() {
    if (_isLoading) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(
              strokeWidth: 3,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 24),
            Text(
              'Loading model...',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
          ],
        ),
      );
    }

    return FadeTransition(
      opacity: _fadeAnimation,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildImageInputSection(),
            if (_selectedImage != null) ...[
              const SizedBox(height: 24),
              _buildImagePreview(),
            ],
            if (_classificationResult != null || _objectDetectionResult != null || _errorMessage != null) ...[
              const SizedBox(height: 24),
              _buildResultsSection(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildImageInputSection() {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.add_photo_alternate,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  'Select Image',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isProcessing ? null : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          if (_isProcessing)
            LinearProgressIndicator(
              backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
            ),
          Stack(
            children: [
              Image.file(
                _selectedImage!,
                height: 300,
                width: double.infinity,
                fit: BoxFit.cover,
              ),
              if (_objectDetectionResult != null)
                Positioned.fill(
                  child: CustomPaint(
                    painter: BoundingBoxPainter(
                      detections: _objectDetectionResult!.detectedObjects,
                      imageSize: const Size(300, 300),
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildResultsSection() {
    if (_errorMessage != null) {
      return Card(
        elevation: 0,
        color: Theme.of(context).colorScheme.errorContainer,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              Icon(
                Icons.error_outline,
                color: Theme.of(context).colorScheme.onErrorContainer,
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  _errorMessage!,
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onErrorContainer,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.check_circle,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  'Results',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                if (_processingTime != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.primaryContainer,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '${_processingTime!.toStringAsFixed(0)}ms',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).colorScheme.onPrimaryContainer,
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            if (_classificationResult != null)
              _buildClassificationResults()
            else if (_objectDetectionResult != null)
              _buildObjectDetectionResults(),
          ],
        ),
      ),
    );
  }

  Widget _buildClassificationResults() {
    return Column(
      children: [
        _buildResultItem(
          _classificationResult!.className,
          _classificationResult!.confidence,
          isPrimary: true,
        ),
        if (_classificationResult!.topK.length > 1) ...[
          const SizedBox(height: 12),
          const Divider(),
          const SizedBox(height: 12),
          Text(
            'Top Predictions',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
            ),
          ),
          const SizedBox(height: 8),
          ...(_classificationResult!.topK.skip(1).take(4).map((result) => Padding(
            padding: const EdgeInsets.only(top: 8),
            child: _buildResultItem(result.className, result.confidence),
          ))),
        ],
      ],
    );
  }

  Widget _buildObjectDetectionResults() {
    final detections = _objectDetectionResult!.detectedObjects;

    if (detections.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Text(
            'No objects detected',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
            ),
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '${detections.length} object${detections.length != 1 ? 's' : ''} detected',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 12),
        ...detections.take(10).map((detection) => Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: _getColorForClass(detection.classIndex),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  detection.className,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
              Text(
                '${(detection.confidence * 100).toStringAsFixed(1)}%',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
            ],
          ),
        )),
      ],
    );
  }

  Widget _buildResultItem(String label, double confidence, {bool isPrimary = false}) {
    return Row(
      children: [
        Expanded(
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              fontWeight: isPrimary ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ),
        const SizedBox(width: 16),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: isPrimary
                ? Theme.of(context).colorScheme.primary
                : Theme.of(context).colorScheme.primaryContainer,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            '${(confidence * 100).toStringAsFixed(1)}%',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.bold,
              color: isPrimary
                  ? Theme.of(context).colorScheme.onPrimary
                  : Theme.of(context).colorScheme.onPrimaryContainer,
            ),
          ),
        ),
      ],
    );
  }

  Color _getColorForClass(int classIndex) {
    final colors = [
      Colors.red,
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.amber,
      Colors.cyan,
      Colors.lime,
    ];
    return colors[classIndex % colors.length];
  }
}

/// Custom painter for drawing bounding boxes over detected objects
class BoundingBoxPainter extends CustomPainter {
  BoundingBoxPainter({
    required this.detections,
    required this.imageSize,
  });

  final List<DetectedObject> detections;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width;
    final scaleY = size.height;

    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];
      final box = detection.boundingBox;

      final left = box.x * scaleX;
      final top = box.y * scaleY;
      final width = box.width * scaleX;
      final height = box.height * scaleY;

      final rect = Rect.fromLTWH(left, top, width, height);
      final color = _getColorForClass(detection.classIndex);

      // Draw bounding box
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;

      canvas.drawRect(rect, boxPaint);

      // Draw label background
      final labelText = '${detection.className} ${(detection.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final labelTop = top > textPainter.height + 8 ? top - textPainter.height - 8 : top + height + 8;
      final labelRect = Rect.fromLTWH(
        left,
        labelTop,
        textPainter.width + 12,
        textPainter.height + 8,
      );

      final labelPaint = Paint()
        ..color = color
        ..style = PaintingStyle.fill;

      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(4)),
        labelPaint,
      );

      textPainter.paint(canvas, Offset(left + 6, labelTop + 4));
    }
  }

  Color _getColorForClass(int classIndex) {
    final colors = [
      Colors.red,
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.amber,
      Colors.cyan,
      Colors.lime,
    ];
    return colors[classIndex % colors.length];
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return detections != oldDelegate.detections || imageSize != oldDelegate.imageSize;
  }
}
