import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_definition.dart';
import '../models/model_registry.dart';
import '../services/model_controller.dart';
import '../ui/widgets/performance_monitor.dart';

/// Unified Model Playground - works with any model type through ModelDefinition
class UnifiedModelPlayground extends StatefulWidget {
  const UnifiedModelPlayground({super.key});

  @override
  State<UnifiedModelPlayground> createState() => _UnifiedModelPlaygroundState();
}

class _UnifiedModelPlaygroundState extends State<UnifiedModelPlayground> {
  // Model state
  List<ModelDefinition>? _availableModels;
  ModelController? _controller;

  // Loading state
  bool _isLoadingModels = true;
  bool _isLoadingModel = false;

  // UI state
  bool _isInputExpanded = true;

  @override
  void initState() {
    super.initState();
    _loadAvailableModels();
  }

  void _onControllerChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  @override
  void dispose() {
    _controller?.removeListener(_onControllerChanged);
    _controller?.dispose();
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
      debugPrint('Failed to load models: $e');
      setState(() {
        _isLoadingModels = false;
      });
    }
  }

  Future<void> _selectModel(ModelDefinition model) async {
    // Remove listener and dispose previous controller (handles camera cleanup)
    final oldController = _controller;
    if (oldController != null) {
      oldController.removeListener(_onControllerChanged);
    }

    setState(() {
      _controller = null; // Clear controller immediately to avoid stale state
      _isLoadingModel = true;
    });

    // Dispose after clearing reference to prevent race conditions
    await oldController?.dispose();

    try {
      final modelPath = await _loadAssetModel(model.assetPath);
      final execuTorchModel = await ExecuTorchModel.load(modelPath);
      final settings = model.createDefaultSettings();

      final controller = await ModelController.create(
        definition: model,
        execuTorchModel: execuTorchModel,
        settings: settings,
      );

      if (mounted) {
        setState(() {
          _controller = controller;
          _controller!.addListener(_onControllerChanged);
          _isLoadingModel = false;
        });
      } else {
        // Widget was unmounted during loading, clean up
        await controller.dispose();
      }
    } catch (e) {
      debugPrint('❌ Failed to load model: $e');
      if (mounted) {
        setState(() {
          _controller = null; // Ensure controller is null on failure
          _isLoadingModel = false;
        });

        // Show helpful error dialog only if the asset file is missing
        final errorString = e.toString();
        if (errorString.contains('Asset not found') ||
            errorString.contains('Unable to load asset')) {
          _showModelNotFoundError(model);
        } else {
          _showModelLoadError(model, errorString);
        }
      }
    }
  }

  void _showModelNotFoundError(ModelDefinition model) {
    // Get export command from model definition
    final exportCommand = model.getExportCommand();
    final specialSetup = model.getSpecialSetupRequirements();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(
              Icons.warning_amber,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(width: 8),
            const Text('Model Not Found'),
          ],
        ),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'The ${model.displayName} model file is missing.',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              Text(
                'Expected file:',
                style: Theme.of(context).textTheme.labelSmall,
              ),
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(top: 4, bottom: 16),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  model.assetPath,
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(fontFamily: 'monospace'),
                ),
              ),
              Text(
                'To export this model, run:',
                style: Theme.of(context).textTheme.labelSmall,
              ),
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(top: 4, bottom: 8),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: SelectableText(
                  'cd example/python\n$exportCommand',
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(fontFamily: 'monospace'),
                ),
              ),
              if (specialSetup != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(
                        Icons.info_outline,
                        size: 16,
                        color: Theme.of(context).colorScheme.onPrimaryContainer,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          specialSetup,
                          style: TextStyle(
                            fontSize: 12,
                            color: Theme.of(
                              context,
                            ).colorScheme.onPrimaryContainer,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showModelLoadError(ModelDefinition model, String error) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(
              Icons.error_outline,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(width: 8),
            const Text('Failed to Load Model'),
          ],
        ),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Failed to load ${model.displayName}',
                style: Theme.of(
                  context,
                ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              Text(
                'Error details:',
                style: Theme.of(context).textTheme.labelSmall,
              ),
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(top: 4),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: SelectableText(
                  error,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontFamily: 'monospace',
                    color: Theme.of(context).colorScheme.error,
                  ),
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Future<String> _loadAssetModel(String assetPath) async {
    try {
      final byteData = await rootBundle.load(assetPath);
      final directory = await getApplicationCacheDirectory();
      final fileName = assetPath.split('/').last;
      final file = File('${directory.path}/$fileName');
      await file.writeAsBytes(byteData.buffer.asUint8List());
      return file.path;
    } catch (e) {
      // Re-throw with clearer error message
      throw Exception('Asset not found: $assetPath');
    }
  }

  void _showSettingsDialog() {
    if (_controller == null) return;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => ListenableBuilder(
        listenable: _controller!,
        builder: (context, _) => _controller!.buildSettingsWidget(context),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Playground'),
        elevation: 0,
        actions: [
          // Settings button - only shown when a model is selected
          if (_controller != null)
            IconButton(
              icon: const Icon(Icons.settings),
              onPressed: _showSettingsDialog,
              tooltip: 'Model Settings',
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
                  child: _controller == null
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
                                    child: _buildResultSection(
                                      isLargeScreen: true,
                                    ),
                                  ),

                                  // Right: Input + Details (40% width)
                                  Expanded(flex: 4, child: _buildSidePanel()),
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
                                  if (!_isInputExpanded && _controller != null)
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
        initialValue: _controller?.definition,
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
            color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.5),
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
    if (_controller == null || !_isInputExpanded) return const SizedBox();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
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
                style: Theme.of(
                  context,
                ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
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
          _controller!.buildInputWidget(
            context: context,
            onInputSelected: (input) => _controller?.processInput(input),
          ),
        ],
      ),
    );
  }

  Widget _buildResultSection({bool isLargeScreen = false}) {
    if (_controller?.isCameraMode ?? false) {
      return _buildCameraSection();
    }

    if (_controller?.isProcessing ?? false) {
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

    final errorMessage = _controller?.errorMessage;
    if (errorMessage != null) {
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
                      errorMessage,
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

    final input = _controller?.currentInput;
    if (input == null) {
      return Center(
        child: Text(
          'Select an input to see results',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );
    }

    final result = _controller?.currentResult;

    if (isLargeScreen) {
      return Container(
        padding: const EdgeInsets.all(16),
        child: Center(
          child: _controller!.buildResultRenderer(
            context: context,
            input: input,
            result: result,
          ),
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          SizedBox(
            height: 400,
            child: _controller!.buildResultRenderer(
              context: context,
              input: input,
              result: result,
            ),
          ),

          if (result != null) _buildDetailsSection(),

          if (_isInputExpanded) const SizedBox(height: 150),
        ],
      ),
    );
  }

  Widget _buildCameraSection() {
    final input = _controller?.currentInput;
    final result = _controller?.currentResult;
    final showPerformance =
        _controller?.settings.showPerformanceOverlay ?? true;
    final performanceMetrics = _controller?.performanceMetrics;

    Widget cameraContent;
    if (input == null) {
      cameraContent = const Center(child: CircularProgressIndicator());
    } else {
      cameraContent = RepaintBoundary(
        child: _controller!.buildResultRenderer(
          context: context,
          input: input,
          result: result,
        ),
      );
    }

    return Stack(
      children: [
        cameraContent,
        if (showPerformance && (performanceMetrics?.hasData ?? false))
          Positioned(
            top: 16,
            right: 16,
            child: _controller!.definition.buildPerformanceMonitor(
              context: context,
              metrics: performanceMetrics!,
              displayMode: PerformanceDisplayMode.overlay,
            ),
          ),
      ],
    );
  }

  Widget _buildDetailsSection() {
    final showPerformance =
        !(_controller?.isCameraMode ?? false) &&
        (_controller?.settings.showPerformanceOverlay ?? true);
    final performanceMetrics = _controller?.performanceMetrics;
    final result = _controller?.currentResult;

    if (result == null) return const SizedBox();

    return Column(
      children: [
        if (showPerformance && (performanceMetrics?.hasData ?? false))
          _controller!.definition.buildPerformanceMonitor(
            context: context,
            metrics: performanceMetrics!,
            displayMode: PerformanceDisplayMode.section,
          ),

        if (showPerformance && (performanceMetrics?.hasData ?? false))
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            child: const Divider(),
          ),

        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
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
                ],
              ),
              const SizedBox(height: 16),
              _controller!.definition.buildResultsDetailsSection(
                context: context,
                result: result,
                processingTime: performanceMetrics?.totalTime,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSidePanel() {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerLow,
        border: Border(
          left: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
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
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _controller!.buildInputWidget(
                    context: context,
                    onInputSelected: (input) =>
                        _controller?.processInput(input),
                  ),
                ],
              ),
            ),

            // Results details section
            if (_controller?.currentResult != null) _buildDetailsSection(),
          ],
        ),
      ),
    );
  }
}
