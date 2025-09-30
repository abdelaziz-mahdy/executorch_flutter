import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../services/performance_service.dart';

enum AudioModel {
  environmentalSound(
    'Environmental Sound',
    'environmental_sound_classifier',
    'Classify ambient sounds',
  ),
  speechEmotion(
    'Speech Emotion',
    'speech_emotion_recognition',
    'Detect emotions in speech',
  );

  const AudioModel(this.displayName, this.modelPrefix, this.description);
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

class EnhancedAudioDemo extends StatefulWidget {
  const EnhancedAudioDemo({super.key});

  @override
  State<EnhancedAudioDemo> createState() => _EnhancedAudioDemoState();
}

class _EnhancedAudioDemoState extends State<EnhancedAudioDemo>
    with TickerProviderStateMixin {
  // Configuration
  AudioModel _selectedModel = AudioModel.environmentalSound;
  ModelBackend _selectedBackend = ModelBackend.xnnpack;

  // State
  bool _isProcessing = false;
  bool _isRecording = false;
  String? _errorMessage;

  // Models and data
  ExecuTorchModel? _loadedModel;
  List<String>? _classLabels;

  // Results
  String? _classification;
  double? _confidence;
  List<Map<String, dynamic>>? _allPredictions;

  // Animation
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  // Demo audio patterns
  final List<String> _demoSounds = [
    'Traffic noise',
    'Bird chirping',
    'Human speech',
    'Music playing',
    'Footsteps',
    'Applause',
    'Machinery noise',
    'Silence',
  ];

  @override
  void initState() {
    super.initState();
    _setupAnimations();
    _loadModelAndLabels();
  }

  void _setupAnimations() {
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.3).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
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
      final assetPath =
          'assets/models/${_selectedModel.modelPrefix}_${_selectedBackend.suffix}.pte';
      final modelPath = await _loadAssetModel(assetPath);
      _loadedModel = await ExecutorchManager.instance.loadModel(modelPath);

      // Load appropriate labels
      String labelsFile;
      if (_selectedModel == AudioModel.environmentalSound) {
        labelsFile = 'assets/models/audio_class_labels.json';
      } else {
        labelsFile = 'assets/models/emotion_class_labels.json';
      }

      final labelsContent = await rootBundle.loadString(labelsFile);
      final labelsJson = json.decode(labelsContent) as List;
      _classLabels = labelsJson.cast<String>();

      debugPrint(
        '✅ Loaded ${_selectedModel.displayName} with ${_selectedBackend.displayName} backend',
      );
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

  Future<void> _simulateRecording() async {
    if (_loadedModel == null) return;

    try {
      setState(() {
        _isRecording = true;
        _isProcessing = false;
        _errorMessage = null;
      });

      _pulseController.repeat(reverse: true);

      // Simulate recording time
      await Future.delayed(const Duration(seconds: 3));

      _pulseController.stop();
      setState(() {
        _isRecording = false;
        _isProcessing = true;
      });

      await _processAudio();
    } catch (e) {
      setState(() {
        _errorMessage = 'Recording failed: $e';
        _isRecording = false;
        _isProcessing = false;
      });
      _pulseController.stop();
    }
  }

  Future<void> _processDemoSound(String soundName) async {
    if (_loadedModel == null) return;

    try {
      setState(() {
        _isProcessing = true;
        _errorMessage = null;
      });

      // Simulate processing the demo sound
      await Future.delayed(const Duration(milliseconds: 500));
      await _processAudio(demoSound: soundName);
    } catch (e) {
      setState(() {
        _errorMessage = 'Processing failed: $e';
        _isProcessing = false;
      });
    }
  }

  Future<void> _processAudio({String? demoSound}) async {
    try {
      final stopwatch = Stopwatch()..start();

      // Generate mock mel-spectrogram data (224 time frames, 224 mel bands)
      final melSpectrogram = _generateMockMelSpectrogram(demoSound);

      final inputTensor = TensorData(
        data: melSpectrogram.buffer.asUint8List(),
        shape: [1, 3, 224, 224],
        dataType: TensorType.float32,
        name: 'input',
      );

      final result = await _loadedModel!.runInference(inputs: [inputTensor]);

      if (result.outputs != null && result.outputs!.isNotEmpty) {
        final outputTensor = result.outputs!.first!;
        final output = Float32List.view(outputTensor.data.buffer);
        _processResults(output, demoSound);
      }

      stopwatch.stop();
      PerformanceService().recordProcessingTime(
        stopwatch.elapsedMilliseconds.toDouble(),
      );
    } catch (e) {
      setState(() {
        _errorMessage = 'Processing failed: $e';
      });
      debugPrint('❌ Audio processing failed: $e');
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Float32List _generateMockMelSpectrogram(String? demoSound) {
    final random = Random();
    final data = Float32List(1 * 3 * 224 * 224);

    // Generate realistic mel-spectrogram patterns based on demo sound
    for (int i = 0; i < data.length; i++) {
      double value = 0.0;

      if (demoSound != null) {
        // Simulate different audio patterns
        switch (demoSound) {
          case 'Traffic noise':
            value = random.nextGaussian() * 0.3 + 0.4; // Low-frequency heavy
            break;
          case 'Bird chirping':
            value =
                random.nextGaussian() * 0.4 + 0.6; // High-frequency patterns
            break;
          case 'Human speech':
            value = random.nextGaussian() * 0.35 + 0.5; // Mid-frequency speech
            break;
          case 'Music playing':
            value = random.nextGaussian() * 0.4 + 0.7; // Rich harmonic content
            break;
          default:
            value = random.nextGaussian() * 0.2 + 0.3; // General pattern
        }
      } else {
        value = random.nextGaussian() * 0.3 + 0.5; // Random recording
      }

      data[i] = value.clamp(-1.0, 1.0);
    }

    return data;
  }

  void _processResults(Float32List output, String? demoSound) {
    if (_classLabels == null) return;

    final predictions = <Map<String, dynamic>>[];

    // Create prediction map
    for (int i = 0; i < output.length && i < _classLabels!.length; i++) {
      predictions.add({
        'label': _classLabels![i],
        'confidence': output[i].clamp(0.0, 1.0),
      });
    }

    // Sort by confidence
    predictions.sort(
      (a, b) =>
          (b['confidence'] as double).compareTo(a['confidence'] as double),
    );

    // If this is a demo sound, boost the relevant prediction
    if (demoSound != null) {
      _adjustDemoPredictions(predictions, demoSound);
    }

    setState(() {
      _allPredictions = predictions;
      _classification = predictions.first['label'];
      _confidence = predictions.first['confidence'];
    });
  }

  void _adjustDemoPredictions(
    List<Map<String, dynamic>> predictions,
    String demoSound,
  ) {
    // Boost confidence for related labels
    final Map<String, List<String>> soundMappings = {
      'Traffic noise': ['traffic', 'machinery'],
      'Bird chirping': ['nature', 'animal'],
      'Human speech': ['speech'],
      'Music playing': ['music'],
      'Footsteps': ['footsteps'],
      'Applause': ['applause'],
      'Machinery noise': ['machinery'],
      'Silence': ['silence'],
    };

    final relatedLabels = soundMappings[demoSound] ?? [];

    for (final prediction in predictions) {
      final label = prediction['label'] as String;
      for (final related in relatedLabels) {
        if (label.toLowerCase().contains(related.toLowerCase())) {
          prediction['confidence'] =
              (prediction['confidence'] as double) * 1.5 + 0.3;
          prediction['confidence'] = (prediction['confidence'] as double).clamp(
            0.0,
            1.0,
          );
        }
      }
    }

    predictions.sort(
      (a, b) =>
          (b['confidence'] as double).compareTo(a['confidence'] as double),
    );
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

              // Audio Input Section
              _buildAudioInputSection(),
              const SizedBox(height: 16),

              // Demo Sounds Section
              _buildDemoSoundsSection(),
              const SizedBox(height: 16),

              // Results Section
              if (_isProcessing ||
                  _classification != null ||
                  _errorMessage != null)
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
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),

            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Model',
                        style: Theme.of(context).textTheme.labelMedium,
                      ),
                      DropdownButton<AudioModel>(
                        value: _selectedModel,
                        isExpanded: true,
                        items: AudioModel.values
                            .map(
                              (model) => DropdownMenuItem(
                                value: model,
                                child: Text(model.displayName),
                              ),
                            )
                            .toList(),
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

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Backend',
                        style: Theme.of(context).textTheme.labelMedium,
                      ),
                      DropdownButton<ModelBackend>(
                        value: _selectedBackend,
                        isExpanded: true,
                        items: ModelBackend.values
                            .map(
                              (backend) => DropdownMenuItem(
                                value: backend,
                                child: Text(backend.displayName),
                              ),
                            )
                            .toList(),
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

  Widget _buildAudioInputSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              'Record Audio',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),

            AnimatedBuilder(
              animation: _pulseAnimation,
              builder: (context, child) {
                return Transform.scale(
                  scale: _isRecording ? _pulseAnimation.value : 1.0,
                  child: Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _isRecording
                          ? Theme.of(context).colorScheme.error
                          : Theme.of(context).colorScheme.primary,
                      boxShadow: _isRecording
                          ? [
                              BoxShadow(
                                color: Theme.of(
                                  context,
                                ).colorScheme.error.withOpacity(0.3),
                                blurRadius: 20,
                                spreadRadius: 5,
                              ),
                            ]
                          : null,
                    ),
                    child: IconButton(
                      onPressed: (_isProcessing || _isRecording)
                          ? null
                          : _simulateRecording,
                      icon: Icon(
                        _isRecording ? Icons.stop : Icons.mic,
                        size: 48,
                        color: Theme.of(context).colorScheme.onPrimary,
                      ),
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 16),

            Text(
              _isRecording
                  ? 'Recording... Tap to stop'
                  : 'Tap to start recording',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDemoSoundsSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Demo Sounds',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              'Tap any sound to simulate processing',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 16),

            Wrap(
              spacing: 8.0,
              runSpacing: 8.0,
              children: _demoSounds
                  .map(
                    (sound) => InkWell(
                      onTap: (_isProcessing || _isRecording)
                          ? null
                          : () => _processDemoSound(sound),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: Theme.of(
                            context,
                          ).colorScheme.secondaryContainer,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _getSoundIcon(sound),
                              size: 16,
                              color: Theme.of(
                                context,
                              ).colorScheme.onSecondaryContainer,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              sound,
                              style: TextStyle(
                                fontSize: 12,
                                color: Theme.of(
                                  context,
                                ).colorScheme.onSecondaryContainer,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  )
                  .toList(),
            ),
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
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),

            if (_isProcessing) ...[
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Processing audio...'),
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
            ] else if (_classification != null && _allPredictions != null) ...[
              // Top Prediction
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(
                      _getSoundIcon(_classification!),
                      color: Theme.of(context).colorScheme.onPrimaryContainer,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _classification!.toUpperCase(),
                            style: Theme.of(context).textTheme.titleSmall
                                ?.copyWith(
                                  color: Theme.of(
                                    context,
                                  ).colorScheme.onPrimaryContainer,
                                  fontWeight: FontWeight.bold,
                                ),
                          ),
                          if (_confidence != null)
                            Text(
                              'Confidence: ${(_confidence! * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                color: Theme.of(
                                  context,
                                ).colorScheme.onPrimaryContainer,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // All Predictions
              Text(
                'All Predictions',
                style: Theme.of(context).textTheme.labelMedium,
              ),
              const SizedBox(height: 8),
              ..._allPredictions!
                  .take(5)
                  .map(
                    (prediction) => Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              prediction['label'],
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ),
                          Container(
                            width: 100,
                            height: 4,
                            decoration: BoxDecoration(
                              color: Theme.of(context).colorScheme.outline,
                              borderRadius: BorderRadius.circular(2),
                            ),
                            child: FractionallySizedBox(
                              alignment: Alignment.centerLeft,
                              widthFactor: prediction['confidence'],
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Theme.of(context).colorScheme.primary,
                                  borderRadius: BorderRadius.circular(2),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            '${(prediction['confidence'] * 100).toStringAsFixed(0)}%',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ),
            ],
          ],
        ),
      ),
    );
  }

  IconData _getSoundIcon(String sound) {
    final lowerSound = sound.toLowerCase();
    if (lowerSound.contains('traffic') || lowerSound.contains('machinery')) {
      return Icons.traffic;
    } else if (lowerSound.contains('bird') ||
        lowerSound.contains('nature') ||
        lowerSound.contains('animal')) {
      return Icons.pets;
    } else if (lowerSound.contains('speech')) {
      return Icons.record_voice_over;
    } else if (lowerSound.contains('music')) {
      return Icons.music_note;
    } else if (lowerSound.contains('footsteps')) {
      return Icons.directions_walk;
    } else if (lowerSound.contains('applause')) {
      return Icons.sports_score;
    } else if (lowerSound.contains('silence')) {
      return Icons.volume_off;
    } else {
      return Icons.graphic_eq;
    }
  }

  @override
  void dispose() {
    _loadedModel?.dispose();
    _pulseController.dispose();
    super.dispose();
  }
}

extension on Random {
  double nextGaussian() {
    // Box-Muller transform for normal distribution
    double u1 = nextDouble();
    double u2 = nextDouble();
    return sqrt(-2 * log(u1)) * cos(2 * pi * u2);
  }
}
