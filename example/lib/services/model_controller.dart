import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import '../models/model_definition.dart';
import '../models/model_input.dart';
import '../models/model_settings.dart';
import '../models/classification_model_settings.dart';
import '../models/yolo_model_settings.dart';
import '../controllers/camera_controller.dart';
import '../controllers/opencv_camera_controller.dart';
import '../controllers/platform_camera_controller.dart';
import '../processors/camera_image_converter.dart';
import '../ui/widgets/performance_monitor.dart';

/// Central controller that owns ALL model state and lifecycle
///
/// Encapsulates:
/// - Model definition, inference engine, settings
/// - Camera lifecycle and frame processing
/// - Processing state (input, result, timing)
/// - Performance tracking
///
/// Playground is a PURE UI layer that renders controller state.
class ModelController extends ChangeNotifier {
  ModelController._({
    required this.definition,
    required this.execuTorchModel,
    required ModelSettings settings,
  }) : _settings = settings;

  final ModelDefinition definition;
  final ExecuTorchModel execuTorchModel;

  // Settings
  ModelSettings _settings;

  // Camera management
  CameraController? _cameraController;
  StreamSubscription<Uint8List>? _frameSubscription;
  bool _isCameraMode = false;

  // Disposal state - prevents notifyListeners after dispose
  bool _isDisposed = false;

  // Processing state
  ModelInput? _currentInput;
  dynamic _currentResult;
  bool _isProcessing = false;
  bool _isProcessingFrame = false;
  String? _errorMessage;

  // Timing metrics
  double? _preprocessingTime;
  double? _inferenceTime;
  double? _postprocessingTime;
  double? _totalTime;

  // Performance tracking (for camera mode)
  final PerformanceTracker _performanceTracker = PerformanceTracker();

  // Getters for UI
  ModelSettings get settings => _settings;
  bool get isCameraMode => _isCameraMode;
  ModelInput? get currentInput => _currentInput;
  dynamic get currentResult => _currentResult;
  bool get isProcessing => _isProcessing;
  String? get errorMessage => _errorMessage;
  double? get preprocessingTime => _preprocessingTime;
  double? get inferenceTime => _inferenceTime;
  double? get postprocessingTime => _postprocessingTime;
  double? get totalTime => _totalTime;
  PerformanceMetrics get performanceMetrics => _isCameraMode
      ? _performanceTracker.toMetrics()
      : PerformanceMetrics(
          preprocessingTime: _preprocessingTime,
          inferenceTime: _inferenceTime,
          postprocessingTime: _postprocessingTime,
          totalTime: _totalTime,
        );

  /// Safe notify that won't throw if disposed
  void _safeNotifyListeners() {
    if (!_isDisposed) {
      notifyListeners();
    }
  }

  /// Create controller with preloaded resources
  static Future<ModelController> create({
    required ModelDefinition definition,
    required ExecuTorchModel execuTorchModel,
    required ModelSettings settings,
  }) async {
    final controller = ModelController._(
      definition: definition,
      execuTorchModel: execuTorchModel,
      settings: settings,
    );

    // Preload labels if available
    try {
      await (definition as dynamic).loadLabels();
    } catch (e) {
      // Not all models have loadLabels
    }

    controller._updateProcessors();
    return controller;
  }

  /// Log when settings change (processors are created on demand in processInput)
  void _updateProcessors() {
    debugPrint(
      '🔄 Settings updated - processors will be recreated on next use',
    );
  }

  /// Update settings and recreate processors
  void updateSettings(ModelSettings newSettings) {
    // Check if camera provider changed (only for models that support camera)
    CameraProvider? oldCameraProvider;
    CameraProvider? newCameraProvider;

    if (_settings is ClassificationModelSettings) {
      oldCameraProvider =
          (_settings as ClassificationModelSettings).cameraProvider;
    } else if (_settings is YoloModelSettings) {
      oldCameraProvider = (_settings as YoloModelSettings).cameraProvider;
    }

    if (newSettings is ClassificationModelSettings) {
      newCameraProvider = newSettings.cameraProvider;
    } else if (newSettings is YoloModelSettings) {
      newCameraProvider = newSettings.cameraProvider;
    }

    _settings = newSettings;
    _updateProcessors();

    // Recreate camera if provider changed
    if (_isCameraMode &&
        oldCameraProvider != null &&
        newCameraProvider != null &&
        oldCameraProvider != newCameraProvider) {
      debugPrint('📷 Camera provider changed, recreating...');
      _recreateCamera();
    }

    _safeNotifyListeners();
  }

  /// Process static input (images)
  Future<void> processInput(ModelInput input) async {
    if (_isDisposed) return;

    _isProcessing = true;
    _currentInput = input;
    _errorMessage = null;
    _safeNotifyListeners();

    try {
      // Detailed timing
      final preprocessStopwatch = Stopwatch()..start();
      final inputProcessor = definition.createInputProcessor(_settings);
      final tensors = await inputProcessor.process(input);
      preprocessStopwatch.stop();
      _preprocessingTime = preprocessStopwatch.elapsedMilliseconds.toDouble();

      final inferenceStopwatch = Stopwatch()..start();
      final outputs = await execuTorchModel.forward(tensors);
      inferenceStopwatch.stop();
      _inferenceTime = inferenceStopwatch.elapsedMilliseconds.toDouble();

      final postprocessStopwatch = Stopwatch()..start();
      final outputProcessor = definition.createOutputProcessor(_settings);
      final result = await outputProcessor.process(outputs);
      postprocessStopwatch.stop();
      _postprocessingTime = postprocessStopwatch.elapsedMilliseconds.toDouble();

      _totalTime = _preprocessingTime! + _inferenceTime! + _postprocessingTime!;
      _currentResult = result;
      _isProcessing = false;

      _safeNotifyListeners();
    } catch (e) {
      debugPrint('❌ Processing failed: $e');
      _errorMessage = 'Processing failed: $e';
      _isProcessing = false;
      _safeNotifyListeners();
    }
  }

  /// Enable camera mode
  Future<void> enableCameraMode() async {
    if (_isCameraMode || _isDisposed) return;

    debugPrint('📷 Enabling camera...');
    _currentInput = null;
    _currentResult = null;
    _performanceTracker.reset();

    try {
      final provider = _getCameraProvider() ?? _getDefaultCameraProvider();
      _cameraController = _createCamera(provider);
      await _cameraController?.start();

      _frameSubscription = _cameraController?.frameStream.listen(
        _processFrameBytes,
      );
      _isCameraMode = true;

      debugPrint('✅ Camera enabled');
      _safeNotifyListeners();
    } catch (e) {
      debugPrint('❌ Camera failed: $e');
      _errorMessage = 'Camera failed: $e';
      _isCameraMode = false;
      _safeNotifyListeners();
      rethrow;
    }
  }

  /// Disable camera mode
  ///
  /// If [silent] is true, doesn't call notifyListeners (used during dispose)
  Future<void> disableCameraMode({bool silent = false}) async {
    if (!_isCameraMode) return;

    debugPrint('📷 Disabling camera...');
    await _frameSubscription?.cancel();
    _frameSubscription = null;
    await _cameraController?.dispose();
    _cameraController = null;
    _isCameraMode = false;

    debugPrint('✅ Camera disabled');
    if (!silent) {
      _safeNotifyListeners();
    }
  }

  /// Toggle camera
  Future<void> toggleCameraMode() async {
    if (_isCameraMode) {
      await disableCameraMode();
    } else {
      await enableCameraMode();
    }
  }

  /// Process camera frames
  Future<void> _processFrameBytes(Uint8List imageBytes) async {
    // Skip if disposed, already processing, or not in camera mode
    if (_isDisposed || _isProcessingFrame || !_isCameraMode) {
      return;
    }

    _isProcessingFrame = true;

    try {
      final totalStopwatch = Stopwatch()..start();
      final liveCameraInput = LiveCameraInput(imageBytes);

      // Preprocessing
      final preprocessStopwatch = Stopwatch()..start();
      final inputProcessor = definition.createInputProcessor(_settings);
      final tensors = await inputProcessor.process(liveCameraInput);
      preprocessStopwatch.stop();
      final preprocessingTime = preprocessStopwatch.elapsedMilliseconds
          .toDouble();

      // Check again in case we were disposed during preprocessing
      if (_isDisposed) return;

      // Inference
      final inferenceStopwatch = Stopwatch()..start();
      final outputs = await execuTorchModel.forward(tensors);
      inferenceStopwatch.stop();
      final inferenceTime = inferenceStopwatch.elapsedMilliseconds.toDouble();

      // Check again in case we were disposed during inference
      if (_isDisposed) return;

      // Postprocessing
      final postprocessStopwatch = Stopwatch()..start();
      final outputProcessor = definition.createOutputProcessor(_settings);
      final result = await outputProcessor.process(outputs);
      postprocessStopwatch.stop();
      final postprocessingTime = postprocessStopwatch.elapsedMilliseconds
          .toDouble();

      totalStopwatch.stop();
      final totalTime = totalStopwatch.elapsedMilliseconds.toDouble();

      // Final check before updating state
      if (_isDisposed) return;

      // Update performance tracker
      _performanceTracker.update(
        preprocessingTime: preprocessingTime,
        inferenceTime: inferenceTime,
        postprocessingTime: postprocessingTime,
        totalTime: totalTime,
      );

      _currentInput = liveCameraInput;
      _currentResult = result;
      _preprocessingTime = preprocessingTime;
      _inferenceTime = inferenceTime;
      _postprocessingTime = postprocessingTime;
      _totalTime = totalTime;

      _safeNotifyListeners();
    } catch (e) {
      // Only log if not disposed (disposed model throws expected errors)
      if (!_isDisposed) {
        debugPrint('❌ Frame processing failed: $e');
      }
    } finally {
      _isProcessingFrame = false;
    }
  }

  /// Recreate camera
  Future<void> _recreateCamera() async {
    if (!_isCameraMode) return;

    await _cameraController?.dispose();
    final provider = _getCameraProvider() ?? _getDefaultCameraProvider();
    _cameraController = _createCamera(provider);
    await _cameraController?.start();
  }

  /// Get camera provider from settings (if model supports camera)
  CameraProvider? _getCameraProvider() {
    if (_settings is ClassificationModelSettings) {
      return (_settings as ClassificationModelSettings).cameraProvider;
    } else if (_settings is YoloModelSettings) {
      return (_settings as YoloModelSettings).cameraProvider;
    }
    return null;
  }

  /// Create camera
  CameraController _createCamera(CameraProvider provider) {
    switch (provider) {
      case CameraProvider.opencv:
        return OpenCVCameraController(
          deviceId: 0,
          processingInterval: const Duration(milliseconds: 100),
        );
      case CameraProvider.platform:
        return PlatformCameraController(
          converter: ImageLibCameraConverter(),
          processingInterval: const Duration(milliseconds: 100),
        );
    }
  }

  /// Get default camera provider
  CameraProvider _getDefaultCameraProvider() {
    if (kIsWeb) return CameraProvider.platform;
    if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
      return CameraProvider.opencv;
    }
    return CameraProvider.platform;
  }

  /// Build settings widget
  Widget buildSettingsWidget(BuildContext context) {
    return definition.buildSettingsWidget(
      context: context,
      settings: _settings,
      onSettingsChanged: updateSettings,
    );
  }

  /// Build result renderer
  Widget buildResultRenderer({
    required BuildContext context,
    required ModelInput input,
    required dynamic result,
  }) {
    return definition.buildResultRenderer(
      context: context,
      input: input,
      result: result,
    );
  }

  /// Build input widget
  Widget buildInputWidget({
    required BuildContext context,
    required Function(ModelInput) onInputSelected,
  }) {
    return definition.buildInputWidget(
      context: context,
      onInputSelected: onInputSelected,
      onCameraModeToggle: toggleCameraMode,
      isCameraMode: _isCameraMode,
    );
  }

  @override
  Future<void> dispose() async {
    // Mark as disposed first to prevent any further operations
    _isDisposed = true;

    // Disable camera silently (don't notify since we're disposing)
    await disableCameraMode(silent: true);
    await execuTorchModel.dispose();
    super.dispose();
  }
}
