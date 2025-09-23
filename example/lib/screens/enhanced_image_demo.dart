import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../services/performance_service.dart';
import 'package:image/image.dart' as img;

enum ImageModel {
  mobileNetV3('MobileNet V3 Small', 'mobilenet_v3_small', 'ImageNet classification'),
  objectDetector('Object Detector', 'simple_object_detector', 'Real-time object detection');

  const ImageModel(this.displayName, this.modelPrefix, this.description);
  final String displayName;
  final String modelPrefix;
  final String description;
}

enum ModelBackend {
  xnnpack('XNNPACK', 'xnnpack', 'CPU optimized, works on all devices'),
  mps('Metal (MPS)', 'mps', 'GPU optimized for Apple devices'),
  coreml('CoreML', 'coreml', 'Neural Engine optimized for Apple devices');

  const ModelBackend(this.displayName, this.suffix, this.description);
  final String displayName;
  final String suffix;
  final String description;
}

class EnhancedImageDemo extends StatefulWidget {
  const EnhancedImageDemo({super.key});

  @override
  State<EnhancedImageDemo> createState() => _EnhancedImageDemoState();
}

class _EnhancedImageDemoState extends State<EnhancedImageDemo> {
  // Configuration options
  ImageModel _selectedModel = ImageModel.mobileNetV3;
  ModelBackend _selectedBackend = ModelBackend.xnnpack;

  // Results
  ClassificationResult? _classificationResult;
  List<String>? _detectedObjects;
  bool _isProcessing = false;
  String? _errorMessage;

  // Models and Processors
  ExecuTorchModel? _loadedModel;
  List<String>? _classLabels;
  ImageNetProcessor? _imageNetProcessor;

  // Image state (replacing BaseProcessorDemoState)
  File? capturedImage;

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<String> _loadAssetModel(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final directory = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${directory.path}/$fileName');

    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }

  Future<void> _loadModelAndLabels() async {
    try {
      if (mounted) {
        setState(() {
          _isProcessing = true;
          _errorMessage = null;
        });
      }

      // Dispose of previous model
      _loadedModel?.dispose();
      _loadedModel = null;

      // Small delay to ensure cleanup completes
      await Future.delayed(const Duration(milliseconds: 100));

      // Load new model from assets to temp storage
      final assetPath = 'assets/models/${_selectedModel.modelPrefix}_${_selectedBackend.suffix}.pte';
      final modelPath = await _loadAssetModel(assetPath);
      _loadedModel = await ExecutorchManager.instance.loadModel(modelPath);

      // Load appropriate labels and create processors
      if (_selectedModel == ImageModel.mobileNetV3) {
        final labelsContent = await rootBundle.loadString('assets/models/imagenet_classes.txt');
        _classLabels = labelsContent.trim().split('\n');

        // Create ImageNet processor with proper config
        _imageNetProcessor = ImageNetProcessor(
          preprocessConfig: const ImagePreprocessConfig(
            targetWidth: 224,
            targetHeight: 224,
            normalizeToFloat: true,
            meanSubtraction: [0.485, 0.456, 0.406],
            standardDeviation: [0.229, 0.224, 0.225],
            cropMode: ImageCropMode.centerCrop,
          ),
          classLabels: _classLabels!,
        );
      } else if (_selectedModel == ImageModel.objectDetector) {
        // Use simple detection labels for demo
        _classLabels = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light'];
        _imageNetProcessor = null; // Object detection doesn't use ImageNet processor
      }

      debugPrint('✅ Loaded ${_selectedModel.displayName} with ${_selectedBackend.displayName} backend');

    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to load model: $e';
        });
      }
      debugPrint('❌ Model loading failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  Future<void> _processImage(File imageFile) async {
    if (_loadedModel == null) return;

    try {
      if (mounted) {
        setState(() {
          _isProcessing = true;
          _errorMessage = null;
        });
      }

      final stopwatch = Stopwatch()..start();
      final imageBytes = await imageFile.readAsBytes();

      if (_selectedModel == ImageModel.mobileNetV3) {
        await _runImageClassification(imageBytes);
      } else if (_selectedModel == ImageModel.objectDetector) {
        await _runObjectDetection(imageBytes);
      }

      stopwatch.stop();
      PerformanceService().recordProcessingTime(stopwatch.elapsedMilliseconds.toDouble());

    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Processing failed: $e';
        });
      }
      debugPrint('❌ Image processing failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  Future<void> _runImageClassification(Uint8List imageBytes) async {
    if (_imageNetProcessor == null || _loadedModel == null) {
      throw Exception('ImageNet processor or model not initialized');
    }

    debugPrint('🔍 Starting image classification...');
    debugPrint('📷 Image size: ${imageBytes.length} bytes');

    try {
      // Use the fixed structured processor interface
      final result = await _imageNetProcessor!.process(imageBytes, _loadedModel!);

      debugPrint('✅ Classification result:');
      debugPrint('   Class: ${result.className}');
      debugPrint('   Confidence: ${(result.confidence * 100).toStringAsFixed(2)}%');
      debugPrint('   Index: ${result.classIndex}');

      if (mounted) {
        setState(() {
          _classificationResult = result;
          _detectedObjects = null;
        });
      }
    } catch (e) {
      debugPrint('❌ Classification failed: $e');
      rethrow;
    }
  }

  Future<void> _runObjectDetection(Uint8List imageBytes) async {
    // Create mock input tensor for object detection (320x320)
    // TODO: Create proper object detection processor when available
    final inputData = Float32List.fromList(List.generate(1 * 3 * 320 * 320, (i) => (i % 256) / 255.0));
    final inputTensor = TensorData(
      data: inputData.buffer.asUint8List(),
      shape: [1, 3, 320, 320],
      dataType: TensorType.float32,
      name: 'input',
    );

    final result = await _loadedModel!.runInference(
      inputs: [inputTensor],
    );

    if (result.outputs != null && result.outputs!.isNotEmpty) {
      if (mounted) {
        setState(() {
          _detectedObjects = ['Mock detected: ${_classLabels?.take(3).join(', ') ?? 'Objects'}'];
          _classificationResult = null;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
            // Model Configuration
            _buildConfigurationPanel(),
            const SizedBox(height: 16),

            // Image Input Section
            _buildImageInputSection(),
            const SizedBox(height: 16),

            // Results Section
            if (_isProcessing || _classificationResult != null || _detectedObjects != null || _errorMessage != null)
              _buildResultsSection(),
          ],
          ),
        ),
      ),
    );
  }

  Widget _buildConfigurationPanel() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Model Configuration',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            // Model Selection
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Model', style: Theme.of(context).textTheme.labelMedium),
                      DropdownButton<ImageModel>(
                        value: _selectedModel,
                        isExpanded: true,
                        items: ImageModel.values.map((model) =>
                          DropdownMenuItem(
                            value: model,
                            child: Text(model.displayName),
                          ),
                        ).toList(),
                        onChanged: (model) {
                          if (model != null) {
                            setState(() {
                              _selectedModel = model;
                            });
                            _loadModelAndLabels();
                          }
                        },
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 16),

                // Backend Selection
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Backend', style: Theme.of(context).textTheme.labelMedium),
                      DropdownButton<ModelBackend>(
                        value: _selectedBackend,
                        isExpanded: true,
                        items: ModelBackend.values.map((backend) =>
                          DropdownMenuItem(
                            value: backend,
                            child: Text(backend.displayName),
                          ),
                        ).toList(),
                        onChanged: (backend) {
                          if (backend != null) {
                            setState(() {
                              _selectedBackend = backend;
                            });
                            _loadModelAndLabels();
                          }
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageInputSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              'Select Image',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isProcessing ? null : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),

            if (capturedImage != null) ...[
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(
                  capturedImage!,
                  height: 200,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultsSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Results',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            if (_isProcessing) ...[
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Processing image...'),
                  ],
                ),
              ),
            ] else if (_errorMessage != null) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.error,
                      color: Theme.of(context).colorScheme.onErrorContainer,
                    ),
                    const SizedBox(width: 8),
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
            ] else if (_classificationResult != null) ...[
              // Image Classification Results
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Classification: ${_classificationResult!.className}',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        color: Theme.of(context).colorScheme.onPrimaryContainer,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Confidence: ${(_classificationResult!.confidence * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onPrimaryContainer,
                      ),
                    ),
                    Text(
                      'Class Index: ${_classificationResult!.classIndex}',
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onPrimaryContainer.withOpacity(0.7),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ] else if (_detectedObjects != null) ...[
              // Object Detection Results
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.secondaryContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Detected Objects:',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        color: Theme.of(context).colorScheme.onSecondaryContainer,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    ..._detectedObjects!.map((obj) => Text(
                      '• $obj',
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onSecondaryContainer,
                      ),
                    )),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: source);

    if (image != null) {
      final file = File(image.path);
      setState(() {
        capturedImage = file;
      });
      await _processImage(file);
    }
  }


  @override
  void dispose() {
    _loadedModel?.dispose();
    super.dispose();
  }
}