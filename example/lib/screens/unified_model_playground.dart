import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_definition.dart';
import '../models/model_registry.dart';

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

  // Input/Result state (generic)
  Object? _input;
  Object? _result;
  double? _processingTime;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadAvailableModels();
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
      final execuTorchModel =
          await ExecutorchManager.instance.loadModel(modelPath);

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
      final startTime = DateTime.now();

      // Step 1: Prepare input (convert to TensorData)
      final tensorInputs = await _selectedModel!.prepareInput(input);

      // Step 2: Run inference
      final inferenceResult = await _loadedExecuTorchModel!.runInference(
        inputs: tensorInputs,
      );

      // Step 3: Process result using the model definition
      final result = await _selectedModel!.processResult(
        input: input,
        inferenceResult: inferenceResult,
      );

      final endTime = DateTime.now();
      final processingTime =
          endTime.difference(startTime).inMilliseconds.toDouble();

      setState(() {
        _result = result;
        _processingTime = processingTime;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Processing failed: $e';
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Playground'),
        elevation: 0,
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
                      : Column(
                          children: [
                            // Input section (always visible as bottom sheet area)
                            _buildInputSection(),

                            // Result display area
                            Expanded(
                              child: _buildResultSection(),
                            ),
                          ],
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
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
          ),
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
    if (_selectedModel == null) return const SizedBox();

    return Container(
      padding: const EdgeInsets.all(16),
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
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Input',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 12),
          _selectedModel!.buildInputWidget(
            context: context,
            onInputSelected: _processInput,
          ),
        ],
      ),
    );
  }

  Widget _buildResultSection() {
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

    return Column(
      children: [
        // Result renderer (image with boxes, etc.)
        Expanded(
          child: _selectedModel!.buildResultRenderer(
            context: context,
            input: _input!,
            result: _result,
          ),
        ),

        // Result details section
        if (_result != null)
          Container(
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
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const Spacer(),
                    if (_processingTime != null)
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
                          '${_processingTime!.toStringAsFixed(0)}ms',
                          style:
                              Theme.of(context).textTheme.bodySmall?.copyWith(
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
                _selectedModel!.buildResultsDetailsSection(
                  context: context,
                  result: _result!,
                  processingTime: _processingTime,
                ),
              ],
            ),
          ),
      ],
    );
  }
}
