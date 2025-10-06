import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_definition.dart';
import '../models/model_registry.dart';
import '../services/processor_preferences.dart';

/// Unified Model Playground - works with any model type through ModelDefinition
class UnifiedModelPlayground extends StatefulWidget {
  const UnifiedModelPlayground({super.key});

  @override
  State<UnifiedModelPlayground> createState() => _UnifiedModelPlaygroundState();
}

class _UnifiedModelPlaygroundState extends State<UnifiedModelPlayground> {
  // Model state
  List<ModelDefinition>? _availableModels;
  ModelDefinition? _selectedModel;
  ExecuTorchModel? _loadedExecuTorchModel;

  // Processing state
  bool _isLoadingModels = true;
  bool _isLoadingModel = false;
  bool _isProcessing = false;

  // UI state
  bool _isInputExpanded = true;
  bool _useOpenCVProcessor = false;

  // Input/Result state (generic)
  Object? _input;
  Object? _result;
  double? _preprocessingTime;
  double? _inferenceTime;
  double? _postprocessingTime;
  double? _totalTime;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadAvailableModels();
    _loadProcessorPreference();
  }

  Future<void> _loadProcessorPreference() async {
    final useOpenCV = await ProcessorPreferences.getUseOpenCV();
    setState(() {
      _useOpenCVProcessor = useOpenCV;
    });
  }

  @override
  void dispose() {
    _loadedExecuTorchModel?.dispose();
    super.dispose();
  }

  Future<void> _loadAvailableModels() async {
    try {
      final models = await ModelRegistry.loadAll();
      setState(() {
        _availableModels = models;
        _isLoadingModels = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load models: $e';
        _isLoadingModels = false;
      });
    }
  }

  Future<void> _selectModel(ModelDefinition model) async {
    setState(() {
      _isLoadingModel = true;
      _errorMessage = null;
      _input = null;
      _result = null;
    });

    try {
      // Dispose previous model
      await _loadedExecuTorchModel?.dispose();

      // Load model asset
      final modelPath = await _loadAssetModel(model.assetPath);

      // Load ExecuTorch model
      final execuTorchModel = await ExecutorchManager.instance.loadModel(
        modelPath,
      );

      setState(() {
        _selectedModel = model;
        _loadedExecuTorchModel = execuTorchModel;
        _isLoadingModel = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load model: $e';
        _isLoadingModel = false;
      });
    }
  }

  Future<String> _loadAssetModel(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final directory = await getApplicationCacheDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${directory.path}/$fileName');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }

  Future<void> _processInput(dynamic input) async {
    if (_loadedExecuTorchModel == null || _selectedModel == null) return;

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
      _input = input;
    });

    try {
      final totalStopwatch = Stopwatch()..start();

      // Step 1: Prepare input (convert to TensorData)
      print('⏱️  Starting preprocessing...');
      final preprocessStopwatch = Stopwatch()..start();
      final tensorInputs = await _selectedModel!.prepareInput(input);
      preprocessStopwatch.stop();
      final preprocessingTime = preprocessStopwatch.elapsedMilliseconds.toDouble();
      print('⏱️  Preprocessing completed: ${preprocessingTime.toStringAsFixed(0)}ms');

      // Step 2: Run inference
      print('⏱️  Starting inference...');
      final inferenceStopwatch = Stopwatch()..start();
      final inferenceResult = await _loadedExecuTorchModel!.runInference(
        inputs: tensorInputs,
      );
      inferenceStopwatch.stop();
      final inferenceTime = inferenceStopwatch.elapsedMilliseconds.toDouble();
      print('⏱️  Inference completed: ${inferenceTime.toStringAsFixed(0)}ms');

      // Step 3: Process result using the model definition
      print('⏱️  Starting postprocessing...');
      final postprocessStopwatch = Stopwatch()..start();
      final result = await _selectedModel!.processResult(
        input: input,
        inferenceResult: inferenceResult,
      );
      postprocessStopwatch.stop();
      final postprocessingTime = postprocessStopwatch.elapsedMilliseconds.toDouble();
      print('⏱️  Postprocessing completed: ${postprocessingTime.toStringAsFixed(0)}ms');

      totalStopwatch.stop();
      final totalTime = totalStopwatch.elapsedMilliseconds.toDouble();
      print('⏱️  Total time: ${totalTime.toStringAsFixed(0)}ms');

      setState(() {
        _result = result;
        _preprocessingTime = preprocessingTime;
        _inferenceTime = inferenceTime;
        _postprocessingTime = postprocessingTime;
        _totalTime = totalTime;
        _isProcessing = false;
      });
    } catch (e, stackTrace) {
      setState(() {
        _errorMessage = 'Processing failed: $e';
        _isProcessing = false;
      });
      debugPrint('Error during processing: $e');
      debugPrint('Stack trace: $stackTrace');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Playground'),
        elevation: 0,
        actions: [
          // Processor toggle
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                _useOpenCVProcessor ? 'OpenCV' : 'Dart',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              Switch(
                value: _useOpenCVProcessor,
                onChanged: (value) async {
                  await ProcessorPreferences.setUseOpenCV(value);
                  setState(() {
                    _useOpenCVProcessor = value;
                  });
                  // Show feedback
                  if (mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(
                          'Switched to ${value ? "OpenCV" : "Dart"} processor',
                        ),
                        duration: const Duration(seconds: 1),
                      ),
                    );
                  }
                },
              ),
              const SizedBox(width: 8),
            ],
          ),
        ],
      ),
      body: _isLoadingModels
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                // Model selector
                _buildModelSelector(),

                // Main content
                Expanded(
                  child: _selectedModel == null
                      ? _buildEmptyState()
                      : LayoutBuilder(
                          builder: (context, constraints) {
                            final isLargeScreen = constraints.maxWidth > 900;

                            if (isLargeScreen) {
                              // Horizontal layout for large screens
                              return Row(
                                children: [
                                  // Left: Result display (60% width)
                                  Expanded(
                                    flex: 6,
                                    child: _buildResultSection(isLargeScreen: true),
                                  ),

                                  // Right: Input + Details (40% width)
                                  Expanded(
                                    flex: 4,
                                    child: _buildSidePanel(),
                                  ),
                                ],
                              );
                            } else {
                              // Vertical layout for mobile/small screens
                              return Stack(
                                children: [
                                  // Result display area (full screen, scrollable)
                                  _buildResultSection(isLargeScreen: false),

                                  // Collapsible input panel at bottom
                                  Positioned(
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    child: _buildInputSection(),
                                  ),

                                  // Toggle button
                                  if (!_isInputExpanded && _selectedModel != null)
                                    Positioned(
                                      right: 16,
                                      bottom: 16,
                                      child: FloatingActionButton(
                                        onPressed: () {
                                          setState(() {
                                            _isInputExpanded = true;
                                          });
                                        },
                                        child: const Icon(Icons.input),
                                      ),
                                    ),
                                ],
                              );
                            }
                          },
                        ),
                ),
              ],
            ),
    );
  }

  Widget _buildModelSelector() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: DropdownButtonFormField<ModelDefinition>(
        value: _selectedModel,
        decoration: InputDecoration(
          labelText: 'Select Model',
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
        items: _availableModels?.map((model) {
          return DropdownMenuItem(
            value: model,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(model.icon, size: 20),
                const SizedBox(width: 12),
                Flexible(
                  fit: FlexFit.loose,
                  child: Text(
                    model.displayName,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          );
        }).toList(),
        onChanged: _isLoadingModel
            ? null
            : (model) {
                if (model != null) _selectModel(model);
              },
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.model_training,
            size: 64,
            color: Theme.of(context).colorScheme.primary.withOpacity(0.5),
          ),
          const SizedBox(height: 16),
          Text(
            'Select a model to get started',
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ],
      ),
    );
  }

  Widget _buildInputSection() {
    if (_selectedModel == null || !_isInputExpanded) return const SizedBox();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.input,
                size: 18,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
              const SizedBox(width: 8),
              Text(
                'Input',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.close, size: 20),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                onPressed: () {
                  setState(() {
                    _isInputExpanded = false;
                  });
                },
                tooltip: 'Hide',
              ),
            ],
          ),
          const SizedBox(height: 8),
          _selectedModel!.buildInputWidget(
            context: context,
            onInputSelected: _processInput,
          ),
        ],
      ),
    );
  }

  Widget _buildResultSection({bool isLargeScreen = false}) {
    if (_isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Processing...'),
          ],
        ),
      );
    }

    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Card(
            color: Theme.of(context).colorScheme.errorContainer,
            child: Padding(
              padding: const EdgeInsets.all(16),
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
          ),
        ),
      );
    }

    if (_input == null) {
      return Center(
        child: Text(
          'Select an input to see results',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );
    }

    // For large screens, just show the image/result
    if (isLargeScreen) {
      return Container(
        padding: const EdgeInsets.all(16),
        child: Center(
          child: _selectedModel!.buildResultRenderer(
            context: context,
            input: _input!,
            result: _result,
          ),
        ),
      );
    }

    // For small screens, show image + details below
    return SingleChildScrollView(
      child: Column(
        children: [
          // Result renderer (image with boxes, etc.)
          SizedBox(
            height: 400,
            child: _selectedModel!.buildResultRenderer(
              context: context,
              input: _input!,
              result: _result,
            ),
          ),

          // Result details section
          if (_result != null) _buildDetailsSection(),

          // Bottom padding when input panel is visible
          if (_isInputExpanded) const SizedBox(height: 150),
        ],
      ),
    );
  }

  Widget _buildDetailsSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border(
          top: BorderSide(
            color: Theme.of(context).colorScheme.outlineVariant,
          ),
        ),
      ),
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
                style: Theme.of(context)
                    .textTheme
                    .titleMedium
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              if (_totalTime != null)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '${_totalTime!.toStringAsFixed(0)}ms',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Theme.of(context)
                              .colorScheme
                              .onPrimaryContainer,
                        ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 16),

          // Timing breakdown
          if (_preprocessingTime != null &&
              _inferenceTime != null &&
              _postprocessingTime != null)
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Performance',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 12),
                _buildTimingRow(
                  context,
                  'Preprocessing',
                  _preprocessingTime!,
                  _totalTime!,
                  Colors.blue,
                ),
                const SizedBox(height: 8),
                _buildTimingRow(
                  context,
                  'Inference',
                  _inferenceTime!,
                  _totalTime!,
                  Colors.green,
                ),
                const SizedBox(height: 8),
                _buildTimingRow(
                  context,
                  'Postprocessing',
                  _postprocessingTime!,
                  _totalTime!,
                  Colors.orange,
                ),
                const SizedBox(height: 16),
                const Divider(),
                const SizedBox(height: 16),
              ],
            ),

          _selectedModel!.buildResultsDetailsSection(
            context: context,
            result: _result!,
            processingTime: _totalTime,
          ),
        ],
      ),
    );
  }

  Widget _buildSidePanel() {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerLow,
        border: Border(
          left: BorderSide(
            color: Theme.of(context).colorScheme.outlineVariant,
          ),
        ),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Input section (always visible on large screens)
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surface,
                border: Border(
                  bottom: BorderSide(
                    color: Theme.of(context).colorScheme.outlineVariant,
                  ),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Input',
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  _selectedModel!.buildInputWidget(
                    context: context,
                    onInputSelected: _processInput,
                  ),
                ],
              ),
            ),

            // Results details section
            if (_result != null)
              Padding(
                padding: const EdgeInsets.all(16),
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
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold),
                        ),
                        const Spacer(),
                        if (_totalTime != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: Theme.of(context)
                                  .colorScheme
                                  .primaryContainer,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              '${_totalTime!.toStringAsFixed(0)}ms',
                              style: Theme.of(context)
                                  .textTheme
                                  .bodySmall
                                  ?.copyWith(
                                    fontWeight: FontWeight.bold,
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onPrimaryContainer,
                                  ),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // Timing breakdown
                    if (_preprocessingTime != null &&
                        _inferenceTime != null &&
                        _postprocessingTime != null)
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Performance',
                            style:
                                Theme.of(context).textTheme.titleSmall?.copyWith(
                                      fontWeight: FontWeight.bold,
                                    ),
                          ),
                          const SizedBox(height: 12),
                          _buildTimingRow(
                            context,
                            'Preprocessing',
                            _preprocessingTime!,
                            _totalTime!,
                            Colors.blue,
                          ),
                          const SizedBox(height: 8),
                          _buildTimingRow(
                            context,
                            'Inference',
                            _inferenceTime!,
                            _totalTime!,
                            Colors.green,
                          ),
                          const SizedBox(height: 8),
                          _buildTimingRow(
                            context,
                            'Postprocessing',
                            _postprocessingTime!,
                            _totalTime!,
                            Colors.orange,
                          ),
                          const SizedBox(height: 16),
                          const Divider(),
                          const SizedBox(height: 16),
                        ],
                      ),

                    _selectedModel!.buildResultsDetailsSection(
                      context: context,
                      result: _result!,
                      processingTime: _totalTime,
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimingRow(
    BuildContext context,
    String label,
    double time,
    double totalTime,
    Color color,
  ) {
    final percentage = (time / totalTime * 100).toStringAsFixed(1);
    final ratio = time / totalTime;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                label,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ),
            Text(
              '${time.toStringAsFixed(0)}ms',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '($percentage%)',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: ratio,
            backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}
