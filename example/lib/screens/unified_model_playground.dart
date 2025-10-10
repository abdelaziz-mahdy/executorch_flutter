import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_definition.dart';
import '../models/model_input.dart';
import '../models/model_registry.dart';
import '../models/model_settings.dart';
import '../services/service_locator.dart';
import '../controllers/camera_controller.dart';
import '../controllers/opencv_camera_controller.dart';
import '../controllers/platform_camera_controller.dart';
import '../processors/camera_image_converter.dart';
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
  ModelDefinition? _selectedModel;
  ExecuTorchModel? _loadedExecuTorchModel;
  ModelSettings? _modelSettings;

  // Processing state
  bool _isLoadingModels = true;
  bool _isLoadingModel = false;
  bool _isProcessing = false;
  bool _isProcessingFrame = false; // Prevent frame processing queue buildup

  // UI state
  bool _isInputExpanded = true;
  bool _isCameraMode = false;

  // Input/Result state (generic)
  ModelInput? _input;
  Object? _result;
  double? _preprocessingTime;
  double? _inferenceTime;
  double? _postprocessingTime;
  double? _totalTime;
  String? _errorMessage;

  // Camera controller from GetIt
  CameraController? _cameraController;
  StreamSubscription<Uint8List>? _frameSubscription;

  // Performance tracking for camera mode
  final PerformanceTracker _performanceTracker = PerformanceTracker();

  @override
  void initState() {
    super.initState();
    _loadAvailableModels();
    // Camera controller will be initialized when camera mode is turned on
  }

  Future<void> _toggleCameraMode() async {
    setState(() {
      _isCameraMode = !_isCameraMode;
      _input = null;
      _result = null;

      // Reset performance tracking when entering camera mode
      if (_isCameraMode) {
        _performanceTracker.reset();
      }
    });

    if (_isCameraMode) {
      // Re-register camera controller if it was unregistered
      if (!getIt.isRegistered<CameraController>()) {
        // Determine camera provider from settings, or fall back to platform default
        final cameraProvider = _modelSettings?.cameraProvider ??
            (Platform.isMacOS || Platform.isWindows || Platform.isLinux
                ? CameraProvider.opencv
                : CameraProvider.platform);

        getIt.registerLazySingleton<CameraController>(() {
          switch (cameraProvider) {
            case CameraProvider.opencv:
              debugPrint('📷 Using OpenCV camera');
              return OpenCVCameraController(
                deviceId: 0,
                processingInterval: const Duration(milliseconds: 100),
              );
            case CameraProvider.platform:
              debugPrint('📷 Using platform camera');
              return PlatformCameraController(
                converter: ImageLibCameraConverter(),
                processingInterval: const Duration(milliseconds: 100),
              );
          }
        });
        debugPrint('📝 Re-registered CameraController in GetIt');
      }

      // Get fresh camera controller from GetIt
      _cameraController ??= getIt<CameraController>();

      // Start camera and subscribe to frames
      try {
        await _cameraController?.start();
        _frameSubscription = _cameraController?.frameStream.listen(_processFrameBytes);
        debugPrint('✅ Camera started and subscribed to frame stream');
      } catch (e) {
        debugPrint('❌ Failed to start camera: $e');
        setState(() {
          _isCameraMode = false;
          _errorMessage = 'Failed to start camera: $e';
        });
      }
    } else {
      // Stop camera, unsubscribe, and dispose to free resources
      await _frameSubscription?.cancel();
      _frameSubscription = null;
      await _cameraController?.dispose();
      _cameraController = null;

      // Unregister from GetIt so a fresh instance is created next time
      if (getIt.isRegistered<CameraController>()) {
        getIt.unregister<CameraController>();
      }

      debugPrint('🛑 Camera disposed and resources freed');
    }
  }


  Future<void> _processFrameBytes(Uint8List imageBytes) async {
    // Skip if already processing to prevent queue buildup
    if (_isProcessingFrame) {
      debugPrint('⏭️ Skipping frame - already processing');
      return;
    }

    if (_selectedModel == null || _loadedExecuTorchModel == null) {
      debugPrint('❌ Model not ready');
      return;
    }

    _isProcessingFrame = true;

    try {
      debugPrint('📦 Processing camera frame: ${imageBytes.length} bytes, hashCode: ${imageBytes.hashCode}');

      // Start total time measurement
      final totalStopwatch = Stopwatch()..start();

      // Create LiveCameraInput for processing and rendering
      final liveCameraInput = LiveCameraInput(imageBytes);
      debugPrint('📍 Created LiveCameraInput with hashCode: ${liveCameraInput.frameBytes.hashCode}');

      // Step 1: Prepare input (preprocessing)
      // Model will read settings from GetIt if needed
      debugPrint('⏱️  Starting preprocessing...');
      final preprocessStopwatch = Stopwatch()..start();
      final tensorInputs = await _selectedModel!.prepareInput(liveCameraInput);
      preprocessStopwatch.stop();
      final preprocessingTime = preprocessStopwatch.elapsedMilliseconds.toDouble();
      debugPrint('⏱️  Preprocessing completed: ${preprocessingTime.toStringAsFixed(0)}ms (${tensorInputs.length} tensors)');

      // Step 2: Run inference
      debugPrint('⏱️  Starting inference...');
      final inferenceStopwatch = Stopwatch()..start();
      final outputs = await _loadedExecuTorchModel!.forward(tensorInputs);
      inferenceStopwatch.stop();
      final inferenceTime = inferenceStopwatch.elapsedMilliseconds.toDouble();
      debugPrint('⏱️  Inference completed: ${inferenceTime.toStringAsFixed(0)}ms');

      // Step 3: Process result (postprocessing)
      debugPrint('⏱️  Starting postprocessing...');
      final postprocessStopwatch = Stopwatch()..start();
      final result = await _selectedModel!.processResult(
        input: liveCameraInput,
        outputs: outputs,
      );
      postprocessStopwatch.stop();
      final postprocessingTime = postprocessStopwatch.elapsedMilliseconds.toDouble();
      debugPrint('⏱️  Postprocessing completed: ${postprocessingTime.toStringAsFixed(0)}ms');

      totalStopwatch.stop();
      final totalTime = totalStopwatch.elapsedMilliseconds.toDouble();
      debugPrint('⏱️  Total time: ${totalTime.toStringAsFixed(0)}ms');

      // Update UI with result - LiveCameraInput contains both processing and display data
      if (mounted) {
        final oldInputHash = _input is LiveCameraInput ? (_input as LiveCameraInput).frameBytes.hashCode : null;

        // Update performance tracker with new frame metrics
        _performanceTracker.update(
          preprocessingTime: preprocessingTime,
          inferenceTime: inferenceTime,
          postprocessingTime: postprocessingTime,
          totalTime: totalTime,
        );

        setState(() {
          _input = liveCameraInput;
          _result = result;
          _preprocessingTime = preprocessingTime;
          _inferenceTime = inferenceTime;
          _postprocessingTime = postprocessingTime;
          _totalTime = totalTime;
        });

        debugPrint('✅ UI updated! Old hash: $oldInputHash, New hash: ${liveCameraInput.frameBytes.hashCode}');
        debugPrint('📊 Frame #${_performanceTracker.frameCount} | Avg Total: ${_performanceTracker.toMetrics().totalTime!.toStringAsFixed(1)}ms | FPS: ${_performanceTracker.fps.toStringAsFixed(1)}');
      }
    } catch (e, stackTrace) {
      debugPrint('❌ Camera frame processing error: $e');
      debugPrint('Stack trace: $stackTrace');
    } finally {
      _isProcessingFrame = false;
      debugPrint('🔓 Frame processing unlocked');
    }
  }

  @override
  void dispose() {
    _frameSubscription?.cancel();
    _loadedExecuTorchModel?.dispose();

    // Dispose camera controller if still active
    if (_cameraController != null) {
      _cameraController?.dispose();
      if (getIt.isRegistered<CameraController>()) {
        getIt.unregister<CameraController>();
      }
    }

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
      final execuTorchModel = await ExecuTorchModel.load(modelPath);

      // Initialize default settings for this model and register in GetIt
      // Models will read their settings from GetIt
      if (getIt.isRegistered<ModelSettings>()) {
        getIt.unregister<ModelSettings>();
      }
      // Create default settings based on model type
      // For now, we'll let the model's buildSettingsWidget create defaults
      // So we don't register anything here - it will be registered when settings dialog opens

      setState(() {
        _selectedModel = model;
        _loadedExecuTorchModel = execuTorchModel;
        _modelSettings = null; // Reset settings when switching models
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
      // Model will read settings from GetIt if needed
      debugPrint('⏱️  Starting preprocessing...');
      final preprocessStopwatch = Stopwatch()..start();
      final tensorInputs = await _selectedModel!.prepareInput(input);
      preprocessStopwatch.stop();
      final preprocessingTime = preprocessStopwatch.elapsedMilliseconds
          .toDouble();
      debugPrint(
        '⏱️  Preprocessing completed: ${preprocessingTime.toStringAsFixed(0)}ms',
      );

      // Step 2: Run inference
      debugPrint('⏱️  Starting inference...');
      final inferenceStopwatch = Stopwatch()..start();
      final outputs = await _loadedExecuTorchModel!.forward(tensorInputs);
      inferenceStopwatch.stop();
      final inferenceTime = inferenceStopwatch.elapsedMilliseconds.toDouble();
      debugPrint(
        '⏱️  Inference completed: ${inferenceTime.toStringAsFixed(0)}ms',
      );

      // Step 3: Process result using the model definition
      debugPrint('⏱️  Starting postprocessing...');
      final postprocessStopwatch = Stopwatch()..start();
      final result = await _selectedModel!.processResult(
        input: input,
        outputs: outputs,
      );
      postprocessStopwatch.stop();
      final postprocessingTime = postprocessStopwatch.elapsedMilliseconds
          .toDouble();
      debugPrint(
        '⏱️  Postprocessing completed: ${postprocessingTime.toStringAsFixed(0)}ms',
      );

      totalStopwatch.stop();
      final totalTime = totalStopwatch.elapsedMilliseconds.toDouble();
      debugPrint('⏱️  Total time: ${totalTime.toStringAsFixed(0)}ms');

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

  void _showSettingsDialog() {
    if (_selectedModel == null) return;

    // Show settings in a modal bottom sheet with StatefulBuilder
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) {
          // Initialize default settings if none exist yet
          // This also registers them in GetIt for models to access
          if (_modelSettings == null && getIt.isRegistered<ModelSettings>()) {
            _modelSettings = getIt<ModelSettings>();
          }

          // Get the settings widget from the model
          final settingsWidget = _selectedModel!.buildSettingsWidget(
            context: context,
            settings: _modelSettings,
            onSettingsChanged: (newSettings) async {
              // Check if camera provider changed while camera is active
              final oldCameraProvider = _modelSettings?.cameraProvider;
              final newCameraProvider = newSettings.cameraProvider;
              final cameraProviderChanged = oldCameraProvider != newCameraProvider;

              // Update both modal state and main state
              setModalState(() {
                _modelSettings = newSettings;
              });
              setState(() {
                _modelSettings = newSettings;
              });

              // Register settings in GetIt for models to access
              if (getIt.isRegistered<ModelSettings>()) {
                getIt.unregister<ModelSettings>();
              }
              getIt.registerSingleton<ModelSettings>(_modelSettings!);

              // Restart camera if provider changed and camera is active
              if (cameraProviderChanged && _isCameraMode) {
                debugPrint('📷 Camera provider changed, restarting camera...');

                // Stop current camera
                await _frameSubscription?.cancel();
                _frameSubscription = null;
                await _cameraController?.dispose();
                _cameraController = null;

                // Unregister old controller
                if (getIt.isRegistered<CameraController>()) {
                  getIt.unregister<CameraController>();
                }

                // Re-toggle camera mode to start with new provider
                // Turn off first
                setState(() {
                  _isCameraMode = false;
                });

                // Turn back on with new provider
                await Future.delayed(const Duration(milliseconds: 100));
                await _toggleCameraMode();
              }
            },
          );

          // If the model has no settings, show a message
          if (settingsWidget == null) {
            // Use mounted check pattern for async context usage
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (!context.mounted) return;
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('This model has no configurable settings'),
                  duration: Duration(seconds: 2),
                ),
              );
            });
            return const SizedBox.shrink();
          }

          return DraggableScrollableSheet(
            initialChildSize: 0.6,
            minChildSize: 0.3,
            maxChildSize: 0.9,
            expand: false,
            builder: (context, scrollController) => Column(
              children: [
                // Header
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    border: Border(
                      bottom: BorderSide(
                        color: Theme.of(context).colorScheme.outlineVariant,
                      ),
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.settings,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(width: 12),
                      Text(
                        'Model Settings',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const Spacer(),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                ),

                // Settings content
                Expanded(
                  child: SingleChildScrollView(
                    controller: scrollController,
                    padding: const EdgeInsets.all(16),
                    child: settingsWidget,
                  ),
                ),
              ],
            ),
          );
        },
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
          if (_selectedModel != null)
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
                                  if (!_isInputExpanded &&
                                      _selectedModel != null)
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
        initialValue: _selectedModel,
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
    if (_selectedModel == null || !_isInputExpanded) return const SizedBox();

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
          _selectedModel!.buildInputWidget(
            context: context,
            onInputSelected: _processInput,
            onCameraModeToggle: _toggleCameraMode,
            isCameraMode: _isCameraMode,
          ),
        ],
      ),
    );
  }

  Widget _buildResultSection({bool isLargeScreen = false}) {
    // Show camera stream processor when in camera mode
    if (_isCameraMode &&
        _loadedExecuTorchModel != null &&
        _selectedModel != null) {
      return _buildCameraSection();
    }

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

  Widget _buildCameraSection() {
    // For camera mode, show the result renderer with performance overlay
    Widget cameraContent;

    if (_input == null) {
      // Camera mode with no results yet - show loading indicator
      cameraContent = const Center(
        child: CircularProgressIndicator(),
      );
    } else {
      // Camera has provided input - render it (with or without inference results)
      cameraContent = RepaintBoundary(
        child: _selectedModel!.buildResultRenderer(
          context: context,
          input: _input!,
          result: _result,
        ),
      );
    }

    // Wrap with performance overlay if enabled for this model
    final showPerformance = _selectedModel?.showPerformanceOverlay ?? true;
    final performanceMetrics = _performanceTracker.toMetrics();

    return Stack(
      children: [
        cameraContent,

        // Performance overlay (top-right corner) - uses model-specific implementation
        if (showPerformance && performanceMetrics.hasData)
          Positioned(
            top: 16,
            right: 16,
            child: _selectedModel!.buildPerformanceMonitor(
              context: context,
              metrics: performanceMetrics,
              displayMode: PerformanceDisplayMode.overlay,
            ),
          ),
      ],
    );
  }

  Widget _buildDetailsSection() {
    // Don't show performance in camera mode - it's already shown as an overlay
    final showPerformance = !_isCameraMode &&
                           (_selectedModel?.showPerformanceOverlay ?? true);
    final performanceMetrics = PerformanceMetrics(
      preprocessingTime: _preprocessingTime,
      inferenceTime: _inferenceTime,
      postprocessingTime: _postprocessingTime,
      totalTime: _totalTime,
    );

    return Column(
      children: [
        // Performance section - uses model-specific implementation
        // Only shown for static images, not camera mode (which uses overlay)
        if (showPerformance && performanceMetrics.hasData)
          _selectedModel!.buildPerformanceMonitor(
            context: context,
            metrics: performanceMetrics,
            displayMode: PerformanceDisplayMode.section,
          ),

        // Divider if performance is shown
        if (showPerformance && performanceMetrics.hasData)
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            child: const Divider(),
          ),

        // Model-specific results details
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
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(fontWeight: FontWeight.bold),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _selectedModel!.buildResultsDetailsSection(
                context: context,
                result: _result!,
                processingTime: _totalTime,
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
                  _selectedModel!.buildInputWidget(
                    context: context,
                    onInputSelected: _processInput,
                    onCameraModeToggle: () {
                      setState(() {
                        _isCameraMode = !_isCameraMode;
                        _input = null;
                        _result = null;
                      });
                    },
                    isCameraMode: _isCameraMode,
                  ),
                ],
              ),
            ),

            // Results details section
            if (_result != null)
              _buildDetailsSection(),
          ],
        ),
      ),
    );
  }
}
