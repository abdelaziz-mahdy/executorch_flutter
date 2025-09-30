import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../services/performance_service.dart';

enum TextModel {
  sentimentAnalysis('Sentiment Analysis', 'sentiment_analysis', 'Analyze emotional tone of text'),
  topicClassification('Topic Classification', 'topic_classification', 'Categorize text by topic');

  const TextModel(this.displayName, this.modelPrefix, this.description);
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

class EnhancedTextDemo extends StatefulWidget {
  const EnhancedTextDemo({super.key});

  @override
  State<EnhancedTextDemo> createState() => _EnhancedTextDemoState();
}

class _EnhancedTextDemoState extends State<EnhancedTextDemo> {
  final TextEditingController _textController = TextEditingController();

  // Configuration
  TextModel _selectedModel = TextModel.sentimentAnalysis;
  ModelBackend _selectedBackend = ModelBackend.xnnpack;

  // State
  bool _isProcessing = false;
  String? _errorMessage;

  // Models and data
  ExecuTorchModel? _loadedModel;
  Map<String, dynamic>? _vocabulary;
  Map<String, dynamic>? _classLabels;

  // Results
  String? _sentiment;
  double? _sentimentConfidence;
  String? _topic;
  double? _topicConfidence;

  // Sample texts for quick testing
  final List<String> _sampleTexts = [
    "I absolutely love this new technology! It's amazing how fast and efficient it is.",
    "The economic outlook for the next quarter looks promising with strong growth indicators.",
    "The football team played exceptionally well in yesterday's championship match.",
    "The new government policy will significantly impact healthcare accessibility.",
    "This movie was terrible. I wasted my time and money watching it.",
    "Apple's latest iPhone features cutting-edge AI capabilities and improved camera technology.",
    "The concert was absolutely incredible! The artist's performance was breathtaking.",
    "Climate change continues to be a major challenge requiring urgent global action.",
  ];

  @override
  void initState() {
    super.initState();
    _loadModelAndData();
  }

  Future<String> _loadAssetModel(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final directory = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${directory.path}/$fileName');

    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }

  Future<void> _loadModelAndData() async {
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

      // Load vocabulary and labels
      final vocabContent = await rootBundle.loadString('assets/models/demo_vocabulary.json');
      _vocabulary = json.decode(vocabContent);

      final labelsContent = await rootBundle.loadString('assets/models/text_class_labels.json');
      _classLabels = json.decode(labelsContent);

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

  Future<void> _processText(String text) async {
    if (_loadedModel == null || text.trim().isEmpty) return;

    try {
      if (mounted) {
        setState(() {
          _isProcessing = true;
          _errorMessage = null;
          _sentiment = null;
          _sentimentConfidence = null;
          _topic = null;
          _topicConfidence = null;
        });
      }

      final stopwatch = Stopwatch()..start();

      // Tokenize text (simplified)
      final tokens = _tokenizeText(text);
      final attentionMask = List.filled(128, 1.0);

      // Create input tensors - replicate data for 3 channels
      final tokenFloats = tokens.map((token) => token.toDouble()).toList();
      final replicatedData = <double>[];
      for (int i = 0; i < 3; i++) {
        replicatedData.addAll(tokenFloats);
      }

      final inputIds = TensorData(
        data: Float32List.fromList(replicatedData).buffer.asUint8List(),
        shape: [1, 3, 128],
        dataType: TensorType.float32,
        name: 'input_ids',
      );

      final inputMask = TensorData(
        data: Float32List.fromList(attentionMask).buffer.asUint8List(),
        shape: [1, 128],
        dataType: TensorType.float32,
        name: 'attention_mask',
      );

      final result = await _loadedModel!.runInference(
        inputs: [inputIds],
      );

      if (result.outputs != null && result.outputs!.isNotEmpty) {
        final outputTensor = result.outputs!.first!;
        final output = Float32List.view(outputTensor.data.buffer);
        _processResults(output);
      }

      stopwatch.stop();
      PerformanceService().recordProcessingTime(stopwatch.elapsedMilliseconds.toDouble());

    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Processing failed: $e';
        });
      }
      debugPrint('❌ Text processing failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  List<int> _tokenizeText(String text) {
    // Simple tokenization using vocabulary
    final words = text.toLowerCase().split(RegExp(r'\W+'));
    final tokens = <int>[];

    for (final word in words) {
      if (word.isEmpty) continue;
      final tokenId = _vocabulary?[word] ?? _vocabulary?['<unk>'] ?? 1;
      tokens.add(tokenId);
    }

    // Pad or truncate to 128 tokens
    while (tokens.length < 128) {
      tokens.add(_vocabulary?['<pad>'] ?? 0);
    }

    return tokens.take(128).toList();
  }

  void _processResults(Float32List output) {
    if (_selectedModel == TextModel.sentimentAnalysis) {
      final sentimentLabels = _classLabels?['sentiment_analysis'] as List<dynamic>? ?? ['negative', 'neutral', 'positive'];

      // Find highest confidence sentiment
      double maxConf = 0.0;
      int maxIndex = 0;
      for (int i = 0; i < output.length && i < sentimentLabels.length; i++) {
        if (output[i] > maxConf) {
          maxConf = output[i];
          maxIndex = i;
        }
      }

      if (mounted) {
        setState(() {
          _sentiment = sentimentLabels[maxIndex];
          _sentimentConfidence = maxConf;
        });
      }

    } else if (_selectedModel == TextModel.topicClassification) {
      final topicLabels = _classLabels?['topic_classification'] as List<dynamic>? ?? ['business', 'technology', 'sports', 'politics', 'entertainment'];

      // Find highest confidence topic
      double maxConf = 0.0;
      int maxIndex = 0;
      for (int i = 0; i < output.length && i < topicLabels.length; i++) {
        if (output[i] > maxConf) {
          maxConf = output[i];
          maxIndex = i;
        }
      }

      if (mounted) {
        setState(() {
          _topic = topicLabels[maxIndex];
          _topicConfidence = maxConf;
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

            // Text Input Section
            _buildTextInputSection(),
            const SizedBox(height: 16),

            // Sample Texts
            _buildSampleTextsSection(),
            const SizedBox(height: 16),

            // Results Section
            if (_isProcessing || _sentiment != null || _topic != null || _errorMessage != null)
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

            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Model', style: Theme.of(context).textTheme.labelMedium),
                      DropdownButton<TextModel>(
                        value: _selectedModel,
                        isExpanded: true,
                        items: TextModel.values.map((model) =>
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
                            _loadModelAndData();
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
                            _loadModelAndData();
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

  Widget _buildTextInputSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Enter Text',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            TextField(
              controller: _textController,
              maxLines: 4,
              decoration: const InputDecoration(
                hintText: 'Type your text here for analysis...',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _isProcessing || _textController.text.trim().isEmpty
                    ? null
                    : () => _processText(_textController.text),
                icon: const Icon(Icons.analytics),
                label: const Text('Analyze Text'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSampleTextsSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Sample Texts',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Tap any sample to analyze it',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 16),

            Wrap(
              spacing: 8.0,
              runSpacing: 8.0,
              children: _sampleTexts.map((text) =>
                InkWell(
                  onTap: _isProcessing ? null : () {
                    _textController.text = text;
                    _processText(text);
                  },
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.secondaryContainer,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      text.length > 50 ? '${text.substring(0, 50)}...' : text,
                      style: TextStyle(
                        fontSize: 12,
                        color: Theme.of(context).colorScheme.onSecondaryContainer,
                      ),
                    ),
                  ),
                ),
              ).toList(),
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
                    Text('Analyzing text...'),
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
            ] else if (_sentiment != null) ...[
              // Sentiment Analysis Results
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: _getSentimentColor(_sentiment!),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _getSentimentIcon(_sentiment!),
                          color: Theme.of(context).colorScheme.onPrimaryContainer,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Sentiment: ${_sentiment!.toUpperCase()}',
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            color: Theme.of(context).colorScheme.onPrimaryContainer,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    if (_sentimentConfidence != null)
                      Text(
                        'Confidence: ${(_sentimentConfidence! * 100).toStringAsFixed(1)}%',
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.onPrimaryContainer,
                        ),
                      ),
                  ],
                ),
              ),
            ] else if (_topic != null) ...[
              // Topic Classification Results
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.secondaryContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _getTopicIcon(_topic!),
                          color: Theme.of(context).colorScheme.onSecondaryContainer,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Topic: ${_topic!.toUpperCase()}',
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            color: Theme.of(context).colorScheme.onSecondaryContainer,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    if (_topicConfidence != null)
                      Text(
                        'Confidence: ${(_topicConfidence! * 100).toStringAsFixed(1)}%',
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.onSecondaryContainer,
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Color _getSentimentColor(String sentiment) {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return Colors.green.withOpacity(0.2);
      case 'negative':
        return Colors.red.withOpacity(0.2);
      default:
        return Theme.of(context).colorScheme.primaryContainer;
    }
  }

  IconData _getSentimentIcon(String sentiment) {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return Icons.sentiment_very_satisfied;
      case 'negative':
        return Icons.sentiment_very_dissatisfied;
      default:
        return Icons.sentiment_neutral;
    }
  }

  IconData _getTopicIcon(String topic) {
    switch (topic.toLowerCase()) {
      case 'business':
        return Icons.business;
      case 'technology':
        return Icons.computer;
      case 'sports':
        return Icons.sports;
      case 'politics':
        return Icons.account_balance;
      case 'entertainment':
        return Icons.movie;
      default:
        return Icons.topic;
    }
  }

  @override
  void dispose() {
    _loadedModel?.dispose();
    _textController.dispose();
    super.dispose();
  }
}