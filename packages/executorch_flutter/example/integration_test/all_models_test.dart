// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

import 'package:executorch_flutter_example/services/model_index_service.dart';
import 'package:executorch_flutter_example/services/model_download_service.dart';

/// Test all models from the models repository using the same approach as the example app.
/// This test uses ModelIndexService to discover models and ModelDownloadService to download them.
///
/// Run with:
///   flutter test integration_test/all_models_test.dart -d macos
///   flutter test integration_test/all_models_test.dart -d ios
///   flutter test integration_test/all_models_test.dart -d linux
///   flutter test integration_test/all_models_test.dart -d windows
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late String currentPlatform;
  late ModelIndex modelIndex;
  final testResults = <ModelTestResult>[];
  final downloadService = ModelDownloadService.instance;

  setUpAll(() async {
    // Detect current platform
    if (Platform.isAndroid) {
      currentPlatform = 'android';
    } else if (Platform.isIOS) {
      currentPlatform = 'ios';
    } else if (Platform.isMacOS) {
      currentPlatform = 'macos';
    } else if (Platform.isLinux) {
      currentPlatform = 'linux';
    } else if (Platform.isWindows) {
      currentPlatform = 'windows';
    } else {
      currentPlatform = 'unknown';
    }

    print('\n${'=' * 70}');
    print('  MODEL COMPATIBILITY TEST');
    print('  Platform: $currentPlatform');
    print('  ExecuTorch Version: ${ModelIndexService.selectedVersion}');
    print('=' * 70);

    // Fetch model index from GitHub
    print('\n  Fetching model index...');
    modelIndex = await ModelIndexService.fetchIndex(forceRefresh: true);
    print('  Found ${modelIndex.models.length} models in index');
    print('  Base URL: ${modelIndex.baseUrl}');
  });

  tearDownAll(() {
    // Print summary
    print('\n${'=' * 70}');
    print('  TEST SUMMARY');
    print('=' * 70);

    final passed = testResults.where((r) => r.passed).length;
    final failed = testResults.where((r) => !r.passed && !r.skipped).length;
    final skipped = testResults.where((r) => r.skipped).length;

    print('\n  Total: ${testResults.length}');
    print('  Passed: $passed');
    print('  Failed: $failed');
    print('  Skipped: $skipped');

    // Group by category
    final byCategory = <String, List<ModelTestResult>>{};
    for (final result in testResults) {
      byCategory.putIfAbsent(result.category, () => []).add(result);
    }

    for (final category in byCategory.keys) {
      print('\n  [$category]');
      for (final result in byCategory[category]!) {
        final status = result.skipped
            ? 'SKIP'
            : (result.passed ? 'PASS' : 'FAIL');
        final icon = result.skipped ? '⏭️' : (result.passed ? '✅' : '❌');
        print('    $icon $status: ${result.modelName}');
        if (!result.passed && !result.skipped && result.error != null) {
          // Truncate error to first line for summary
          final errorLine = result.error!.split('\n').first;
          final truncated = errorLine.length > 60
              ? '${errorLine.substring(0, 60)}...'
              : errorLine;
          print('       Error: $truncated');
        }
      }
    }

    // Print failed models with full error
    final failedResults = testResults.where((r) => !r.passed && !r.skipped);
    if (failedResults.isNotEmpty) {
      print('\n${'=' * 70}');
      print('  FAILED MODEL DETAILS');
      print('=' * 70);
      for (final result in failedResults) {
        print('\n  ${result.modelName}:');
        print('    Phase: ${result.failurePhase}');
        print('    Error: ${result.error}');
      }
    }

    print('\n${'=' * 70}\n');
  });

  group('Model Compatibility Tests', () {
    testWidgets('Test all models from index', (tester) async {
      // Get models for current platform
      final platformModels = modelIndex.getModelsForPlatform(currentPlatform);
      print('\n  Models for $currentPlatform: ${platformModels.length}');

      for (final modelEntry in modelIndex.models) {
        final result = ModelTestResult(
          modelName: modelEntry.name,
          category: modelEntry.category,
          backend: modelEntry.backend,
        );

        // Check whether this model can run in the current environment.
        final skipReason = _skipReasonFor(modelEntry, currentPlatform);
        if (skipReason != null) {
          result.skipped = true;
          result.skipReason = skipReason;
          testResults.add(result);
          print('\n⏭️  SKIP: ${modelEntry.name}');
          print('   Reason: $skipReason');
          continue;
        }

        print('\n🧪 Testing: ${modelEntry.name}');
        print('   Category: ${modelEntry.category}');
        print('   Backend: ${modelEntry.backend}');
        print('   Size: ${modelEntry.sizeMB.toStringAsFixed(1)} MB');

        ExecuTorchModel? model;

        try {
          // Phase 1: Download model using ModelDownloadService
          result.failurePhase = 'download';
          print('   📥 Downloading...');

          final downloadInfo = await downloadService.downloadModel(
            modelName: modelEntry.name,
            remoteUrl: modelEntry.remoteUrl,
            expectedHash: modelEntry.hash,
            onProgress: (progress, received, total) {
              // Progress callback - could print dots for long downloads
            },
          );

          if (downloadInfo.state != ModelDownloadState.downloaded) {
            throw Exception('Download failed: ${downloadInfo.errorMessage}');
          }

          print('   ✓ Downloaded: ${downloadInfo.localPath ?? "in memory"}');

          // Phase 2: Load model
          result.failurePhase = 'load';
          print('   📦 Loading model...');

          // Load from file path if available, otherwise from bytes
          if (downloadInfo.localPath != null) {
            model = await ExecuTorchModel.load(downloadInfo.localPath!);
          } else if (downloadInfo.bytes != null) {
            model = await ExecuTorchModel.loadFromBytes(downloadInfo.bytes!);
          } else {
            throw Exception('No model data available after download');
          }
          print('   ✓ Loaded model: ${modelEntry.name}');

          // Phase 3: Create test input
          result.failurePhase = 'create_input';
          print('   🔧 Creating test input...');
          final inputSize = modelEntry.inputSize ?? 224;
          final inputTensor = _createTestInput(inputSize);
          print('   ✓ Input tensor: shape=${inputTensor.shape}');

          // Phase 4: Run inference
          result.failurePhase = 'inference';
          print('   🚀 Running inference...');
          final stopwatch = Stopwatch()..start();
          final outputs = await model.forward([inputTensor]);
          stopwatch.stop();
          print(
            '   ✓ Inference: ${outputs.length} outputs in ${stopwatch.elapsedMilliseconds}ms',
          );

          // Log output shapes
          for (int i = 0; i < outputs.length; i++) {
            print(
              '     Output[$i]: shape=${outputs[i].shape}, dtype=${outputs[i].dataType}',
            );
          }

          // Phase 5: Validate outputs
          result.failurePhase = 'validate';
          if (outputs.isEmpty) {
            throw Exception('No outputs returned from model');
          }

          // Check output data is not empty/all zeros
          final firstOutput = outputs[0];
          if (firstOutput.data.isEmpty) {
            throw Exception('Output tensor data is empty');
          }

          result.passed = true;
          result.inferenceTimeMs = stopwatch.elapsedMilliseconds.toDouble();
          print('   ✅ PASSED');
        } catch (e, stackTrace) {
          result.passed = false;
          result.error = e.toString();
          result.stackTrace = stackTrace.toString();
          print('   ❌ FAILED at ${result.failurePhase}: $e');
        } finally {
          // Cleanup
          if (model != null) {
            try {
              await model.dispose();
            } catch (e) {
              print('   ⚠️  Dispose error: $e');
            }
          }
          testResults.add(result);
        }
      }

      // Fail the test if any model failed (non-skipped)
      final failedModels = testResults
          .where((r) => !r.passed && !r.skipped)
          .toList();
      expect(
        failedModels,
        isEmpty,
        reason:
            '${failedModels.length} model(s) failed: ${failedModels.map((r) => '${r.modelName} (${r.failurePhase}: ${r.error})').join(', ')}',
      );
    });
  });
}

/// Whether the suite is running under CI.
///
/// Set via `--dart-define=CI=true` in the GitHub Actions workflow. Headless CI
/// runners have no usable GPU / Neural Engine, so GPU/NPU backends (CoreML,
/// MPS, Vulkan) cannot initialize there — only XNNPACK (CPU) is exercised.
const bool _isCI = bool.fromEnvironment('CI', defaultValue: false);

/// Platforms each backend can actually run on.
const Map<String, Set<String>> _backendPlatforms = {
  'xnnpack': {'android', 'ios', 'macos', 'linux', 'windows', 'web'},
  'coreml': {'ios', 'macos'},
  'mps': {'ios', 'macos'},
  'vulkan': {'android', 'ios', 'macos', 'linux', 'windows'},
};

/// Returns a human-readable reason to skip [entry] on [platform], or `null` if
/// the model should be tested.
///
/// The model index does not (yet) carry per-model `platforms`, so support is
/// derived from the backend. We additionally skip combinations that are known
/// to be unavailable or unstable in automated runs.
String? _skipReasonFor(ModelIndexEntry entry, String platform) {
  final supported = _backendPlatforms[entry.backend];
  if (supported == null) {
    return 'unknown backend "${entry.backend}"';
  }
  if (!supported.contains(platform)) {
    return '${entry.backend} backend not supported on $platform';
  }
  // yolo-pose / yolo-face Vulkan variants crash the GPU driver during
  // inference: their pose/keypoint output tensors exceed Vulkan limits
  // (same class of issue as oversized YOLO anchor tensors).
  if (entry.backend == 'vulkan' &&
      (entry.category.contains('pose') || entry.category.contains('face'))) {
    return 'vulkan ${entry.category} models crash the driver (known limitation)';
  }
  // Headless CI runners cannot initialize GPU/NPU backends.
  if (_isCI && entry.backend != 'xnnpack') {
    return '${entry.backend} backend skipped in headless CI (no GPU/NPU)';
  }
  return null;
}

/// Create a test input tensor
TensorData _createTestInput(int inputSize) {
  // Create a simple test pattern (gradient)
  final floatCount = 1 * 3 * inputSize * inputSize;
  final floats = Float32List(floatCount);

  // Fill with normalized values (0 to 1)
  for (int i = 0; i < floatCount; i++) {
    floats[i] = (i % 256) / 255.0;
  }

  return TensorData(
    shape: [1, 3, inputSize, inputSize].cast<int?>(),
    dataType: TensorType.float32,
    data: floats.buffer.asUint8List(),
    name: 'input',
  );
}

/// Model test result
class ModelTestResult {
  final String modelName;
  final String category;
  final String backend;
  bool passed = false;
  bool skipped = false;
  String? skipReason;
  String? failurePhase;
  String? error;
  String? stackTrace;
  double? inferenceTimeMs;

  ModelTestResult({
    required this.modelName,
    required this.category,
    required this.backend,
  });
}
