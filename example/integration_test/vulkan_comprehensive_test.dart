// ignore_for_file: avoid_print

/// Comprehensive Vulkan diagnostic to isolate PowerVR failures.
///
/// Tests ALL combinations of:
///   - FP32 vs FP16
///   - With bias vs without bias
///   - Standard conv, pointwise, depthwise
///   - Progressive MobileNet layer slices
///
/// Produces a shareable markdown report at the end.
///
/// Run:
///   cd example
///   flutter test integration_test/vulkan_comprehensive_test.dart -d DEVICE

@Timeout(Duration(minutes: 15))
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Result of running one test variant.
class TestResult {
  final String name;
  final String dtype;
  final String backend;
  final double? mean;
  final double? min;
  final double? max;
  final int? nanCount;
  final int? infCount;
  final int totalElements;
  final List<double>? first8;
  final String? error;
  final int? inferenceMs;
  final List<int?>? outputShape;

  // Element-wise comparison data (populated when comparing VK vs XN)
  double? maxAbsDiff;
  double? avgAbsDiff;
  int? maxDiffIdx;
  double? xnAtMaxDiff;
  double? vkAtMaxDiff;

  // CPU reference from manifest
  double? cpuRefMean;

  TestResult({
    required this.name,
    required this.dtype,
    required this.backend,
    this.mean,
    this.min,
    this.max,
    this.nanCount,
    this.infCount,
    this.totalElements = 0,
    this.first8,
    this.error,
    this.inferenceMs,
    this.outputShape,
    this.cpuRefMean,
  });
}

/// All collected results for the summary table.
final List<TestResult> allResults = [];

/// CPU reference data from manifest.
final Map<String, double> cpuRefMeans = {};

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Comprehensive Vulkan Diagnostic', () {
    Map<String, dynamic>? manifest;

    setUpAll(() async {
      try {
        final jsonStr = await rootBundle.loadString(
          'assets/debug_models/v2_manifest.json',
        );
        manifest = json.decode(jsonStr) as Map<String, dynamic>;
        final tests = manifest!['tests'] as List;
        print('Loaded v2_manifest.json with ${tests.length} test configs');

        // Extract CPU reference means
        for (final t in tests) {
          final name = t['name'] as String;
          final dtype = t['dtype'] as String;
          final cpuRef = t['cpu_ref'] as Map<String, dynamic>?;
          if (cpuRef != null) {
            final mean = (cpuRef['mean'] as num?)?.toDouble();
            if (mean != null) {
              cpuRefMeans['${name}_$dtype'] = mean;
            }
          }
        }
      } catch (e) {
        print('WARNING: Could not load v2_manifest.json: $e');
        print('Run comprehensive_diagnostic.py first to export models.');
      }
    });

    // =====================================================================
    // PART 1: Individual conv variants
    // =====================================================================
    final convTests = [
      _TestConfig('conv3x3_nobias', [1, 3, 8, 8], 2.7),
      _TestConfig('conv3x3_bias', [1, 3, 8, 8], 3.2),
      _TestConfig('conv3x3_bias32ch', [1, 3, 8, 8], 3.2),
      _TestConfig('pw_nobias', [1, 3, 8, 8], 0.3),
      _TestConfig('pw_bias', [1, 3, 8, 8], 0.8),
      _TestConfig('dw_nobias', [1, 4, 8, 8], null),
      _TestConfig('dw_bias', [1, 4, 8, 8], null),
      _TestConfig('add_bias_only', [1, 4, 8, 8], 1.5),
    ];

    for (final tc in convTests) {
      for (final dtype in ['FP32', 'FP16']) {
        testWidgets('${tc.name} $dtype', (tester) async {
          print('\n--- ${tc.name} ($dtype) ---');

          final input = _createOnesInput(tc.inputShape);

          // Run Vulkan
          final fpSuffix = dtype == 'FP16' ? '_fp16' : '_fp32';
          final vkAsset =
              'assets/debug_models/v2_${tc.name}${fpSuffix}_vulkan.pte';
          await _runAndRecord(vkAsset, input, tc.name, dtype, 'Vulkan');

          // Run XNNPACK (FP32 only)
          if (dtype == 'FP32') {
            final xnAsset =
                'assets/debug_models/v2_${tc.name}_fp32_xnnpack.pte';
            await _runAndRecord(xnAsset, input, tc.name, dtype, 'XNNPACK');
          }
        });
      }
    }

    // =====================================================================
    // PART 2: Progressive MobileNet slices
    // =====================================================================
    for (final nLayers in [1, 2, 3, 4]) {
      for (final dtype in ['FP32', 'FP16']) {
        testWidgets('mobilenet_slice_$nLayers $dtype', (tester) async {
          final name = 'mobilenet_slice_$nLayers';
          print('\n--- $name ($dtype) ---');

          final input = _createOnesInput([1, 3, 224, 224]);

          final fpSuffix = dtype == 'FP16' ? '_fp16' : '_fp32';
          final vkAsset =
              'assets/debug_models/v2_$name${fpSuffix}_vulkan.pte';
          await _runAndRecord(vkAsset, input, name, dtype, 'Vulkan');

          if (dtype == 'FP32') {
            final xnAsset =
                'assets/debug_models/v2_${name}_fp32_xnnpack.pte';
            await _runAndRecord(xnAsset, input, name, dtype, 'XNNPACK');
          }
        });
      }
    }

    // =====================================================================
    // ELEMENT-WISE COMPARISON PASS
    // =====================================================================
    testWidgets('Element-wise comparison', (tester) async {
      print('\n--- Running element-wise comparisons ---');

      final testNames = <String>{};
      for (final r in allResults) {
        testNames.add(r.name);
      }

      for (final name in testNames) {
        // Compare FP32 VK vs XN
        final vk = allResults
            .where(
              (r) =>
                  r.name == name &&
                  r.dtype == 'FP32' &&
                  r.backend == 'Vulkan' &&
                  r.error == null,
            )
            .firstOrNull;
        final xn = allResults
            .where(
              (r) =>
                  r.name == name &&
                  r.dtype == 'FP32' &&
                  r.backend == 'XNNPACK' &&
                  r.error == null,
            )
            .firstOrNull;

        if (vk != null && xn != null && vk.first8 != null && xn.first8 != null) {
          // We only have first8 stored; compute element-wise diff from those
          double maxDiff = 0;
          int maxIdx = 0;
          double sumDiff = 0;
          final n = math.min(vk.first8!.length, xn.first8!.length);
          for (int i = 0; i < n; i++) {
            final diff = (vk.first8![i] - xn.first8![i]).abs();
            sumDiff += diff;
            if (diff > maxDiff) {
              maxDiff = diff;
              maxIdx = i;
            }
          }
          vk.maxAbsDiff = maxDiff;
          vk.avgAbsDiff = n > 0 ? sumDiff / n : 0;
          vk.maxDiffIdx = maxIdx;
          vk.xnAtMaxDiff = n > maxIdx ? xn.first8![maxIdx] : null;
          vk.vkAtMaxDiff = n > maxIdx ? vk.first8![maxIdx] : null;
        }
      }
    });

    // =====================================================================
    // SHAREABLE REPORT
    // =====================================================================
    testWidgets('GENERATE REPORT', (tester) async {
      final report = StringBuffer();

      report.writeln('');
      report.writeln('');
      report.writeln('====== REPORT START (copy everything below) ======');
      report.writeln('');
      report.writeln('# Vulkan PowerVR Diagnostic Report');
      report.writeln('');
      report.writeln(
        'Device: Pixel 10 Pro (PowerVR D-Series DXT-48-1536 MC1)',
      );
      report.writeln(
        'Date: ${DateTime.now().toIso8601String().split("T").first}',
      );
      report.writeln('ExecuTorch: main branch (source build)');
      report.writeln(
        'Test: vulkan_comprehensive_test.dart',
      );
      report.writeln('');

      // ------------------------------------------------------------------
      // Section 1: Summary Table
      // ------------------------------------------------------------------
      report.writeln('## 1. FP32 Results (Vulkan vs XNNPACK)');
      report.writeln('');
      report.writeln(
        '| Test | VK Mean | XN Mean | CPU Ref | Mean Diff | Max Diff (first8) | Status |',
      );
      report.writeln(
        '|------|---------|---------|---------|-----------|-------------------|--------|',
      );

      final testNames = <String>[];
      for (final r in allResults) {
        if (!testNames.contains(r.name)) testNames.add(r.name);
      }

      for (final name in testNames) {
        final vk = _findResult(name, 'FP32', 'Vulkan');
        final xn = _findResult(name, 'FP32', 'XNNPACK');
        final cpuRef = cpuRefMeans['${name}_FP32'];

        final vkMean = _fmtMean(vk);
        final xnMean = _fmtMean(xn);
        final cpuStr = cpuRef?.toStringAsFixed(4) ?? '-';

        String meanDiff = '-';
        String maxDiff = '-';
        String status = _getStatus(vk, xn);

        if (vk != null &&
            vk.error == null &&
            xn != null &&
            xn.error == null) {
          meanDiff =
              ((vk.mean ?? 0) - (xn.mean ?? 0)).abs().toStringAsFixed(6);
          maxDiff = vk.maxAbsDiff?.toStringAsFixed(6) ?? '-';
        }

        report.writeln(
          '| $name | $vkMean | $xnMean | $cpuStr | $meanDiff | $maxDiff | $status |',
        );
      }

      // ------------------------------------------------------------------
      // Section 2: FP16 Results
      // ------------------------------------------------------------------
      report.writeln('');
      report.writeln('## 2. FP16 Results (Vulkan only)');
      report.writeln('');
      report.writeln(
        '| Test | VK Mean | CPU Ref | NaN | Inf | Diff vs CPU | Status |',
      );
      report.writeln(
        '|------|---------|---------|-----|-----|-------------|--------|',
      );

      for (final name in testNames) {
        final vk = _findResult(name, 'FP16', 'Vulkan');
        final cpuRef = cpuRefMeans['${name}_FP16'] ?? cpuRefMeans['${name}_FP32'];

        final vkMean = _fmtMean(vk);
        final cpuStr = cpuRef?.toStringAsFixed(4) ?? '-';
        final nanStr = '${vk?.nanCount ?? 0}';
        final infStr = '${vk?.infCount ?? 0}';

        String diffVsCpu = '-';
        String status;

        if (vk == null || vk.error != null) {
          status = 'FAIL';
        } else if ((vk.nanCount ?? 0) > 0) {
          status = 'NaN';
        } else if ((vk.infCount ?? 0) > 0) {
          status = 'Inf';
        } else if (cpuRef != null) {
          final diff = ((vk.mean ?? 0) - cpuRef).abs();
          diffVsCpu = diff.toStringAsFixed(6);
          if (diff < 0.01) {
            status = 'MATCH';
          } else if (diff < 0.5) {
            status = 'CLOSE';
          } else {
            status = 'DIVERGED';
          }
        } else {
          status = 'NO_REF';
        }

        report.writeln(
          '| $name | $vkMean | $cpuStr | $nanStr | $infStr | $diffVsCpu | $status |',
        );
      }

      // ------------------------------------------------------------------
      // Section 3: Element-wise first8 comparison
      // ------------------------------------------------------------------
      report.writeln('');
      report.writeln('## 3. Element-wise Values (first 8 elements, FP32)');
      report.writeln('');
      report.writeln(
        '| Test | Backend | [0] | [1] | [2] | [3] | [4] | [5] | [6] | [7] |',
      );
      report.writeln(
        '|------|---------|-----|-----|-----|-----|-----|-----|-----|-----|',
      );

      for (final name in testNames) {
        for (final backend in ['Vulkan', 'XNNPACK']) {
          final r = _findResult(name, 'FP32', backend);
          if (r == null || r.first8 == null || r.error != null) continue;
          final vals =
              r.first8!.map((v) => v.toStringAsFixed(4)).toList();
          while (vals.length < 8) {
            vals.add('-');
          }
          report.writeln(
            '| $name | $backend | ${vals.join(" | ")} |',
          );
        }
      }

      // ------------------------------------------------------------------
      // Section 4: Bias analysis
      // ------------------------------------------------------------------
      report.writeln('');
      report.writeln('## 4. Bias Analysis');
      report.writeln('');

      // Compare nobias vs bias to check if bias is being applied
      final biasAnalysis = <String, Map<String, double?>>{};
      for (final suffix in ['nobias', 'bias']) {
        for (final prefix in ['conv3x3', 'pw', 'dw']) {
          final name = '${prefix}_$suffix';
          final vk = _findResult(name, 'FP32', 'Vulkan');
          final xn = _findResult(name, 'FP32', 'XNNPACK');
          biasAnalysis[name] = {
            'vk': vk?.mean,
            'xn': xn?.mean,
          };
        }
      }

      report.writeln('Expected: bias=0.5 should add 0.5 to the nobias output.');
      report.writeln('');
      report.writeln(
        '| Conv Type | VK NoBias | VK Bias | VK Diff | XN NoBias | XN Bias | XN Diff | Bias Applied? |',
      );
      report.writeln(
        '|-----------|-----------|---------|---------|-----------|---------|---------|---------------|',
      );

      for (final prefix in ['conv3x3', 'pw', 'dw']) {
        final nbVk = biasAnalysis['${prefix}_nobias']?['vk'];
        final bVk = biasAnalysis['${prefix}_bias']?['vk'];
        final nbXn = biasAnalysis['${prefix}_nobias']?['xn'];
        final bXn = biasAnalysis['${prefix}_bias']?['xn'];

        final vkDiff =
            (nbVk != null && bVk != null) ? (bVk - nbVk) : null;
        final xnDiff =
            (nbXn != null && bXn != null) ? (bXn - nbXn) : null;

        final vkBiasApplied = vkDiff != null && vkDiff.abs() > 0.1;
        final xnBiasApplied = xnDiff != null && xnDiff.abs() > 0.1;

        String biasStatus;
        if (vkBiasApplied && xnBiasApplied) {
          biasStatus = 'YES (both)';
        } else if (!vkBiasApplied && xnBiasApplied) {
          biasStatus = 'NO (VK drops bias!)';
        } else if (vkBiasApplied && !xnBiasApplied) {
          biasStatus = 'VK only (XN missing?)';
        } else {
          biasStatus = 'NO (both missing)';
        }

        report.writeln(
          '| $prefix | '
          '${nbVk?.toStringAsFixed(4) ?? "-"} | '
          '${bVk?.toStringAsFixed(4) ?? "-"} | '
          '${vkDiff?.toStringAsFixed(4) ?? "-"} | '
          '${nbXn?.toStringAsFixed(4) ?? "-"} | '
          '${bXn?.toStringAsFixed(4) ?? "-"} | '
          '${xnDiff?.toStringAsFixed(4) ?? "-"} | '
          '$biasStatus |',
        );
      }

      // add_bias_only check
      final addBiasVk = _findResult('add_bias_only', 'FP32', 'Vulkan');
      final addBiasXn = _findResult('add_bias_only', 'FP32', 'XNNPACK');
      report.writeln('');
      report.writeln(
        'add_bias_only (aten.add, no conv): VK=${_fmtMean(addBiasVk)}, '
        'XN=${_fmtMean(addBiasXn)} - '
        '${addBiasVk?.mean != null && (addBiasVk!.mean! - 1.5).abs() < 0.01 ? "CORRECT" : "WRONG"}',
      );

      // ------------------------------------------------------------------
      // Section 5: Progressive MobileNet
      // ------------------------------------------------------------------
      report.writeln('');
      report.writeln('## 5. Progressive MobileNet V3 Small');
      report.writeln('');
      report.writeln(
        '| Layers | FP32 VK Mean | FP32 XN Mean | FP32 Diff | FP16 VK Mean | FP16 NaN | FP16 Inf |',
      );
      report.writeln(
        '|--------|--------------|--------------|-----------|--------------|----------|----------|',
      );

      for (final n in [1, 2, 3, 4]) {
        final name = 'mobilenet_slice_$n';
        final vk32 = _findResult(name, 'FP32', 'Vulkan');
        final xn32 = _findResult(name, 'FP32', 'XNNPACK');
        final vk16 = _findResult(name, 'FP16', 'Vulkan');

        final fp32Diff = (vk32?.mean != null && xn32?.mean != null)
            ? ((vk32!.mean! - xn32!.mean!).abs().toStringAsFixed(6))
            : '-';

        report.writeln(
          '| 0-${n - 1} | '
          '${_fmtMean(vk32)} | '
          '${_fmtMean(xn32)} | '
          '$fp32Diff | '
          '${_fmtMean(vk16)} | '
          '${vk16?.nanCount ?? 0} | '
          '${vk16?.infCount ?? 0} |',
        );
      }

      // ------------------------------------------------------------------
      // Section 6: Inference timing
      // ------------------------------------------------------------------
      report.writeln('');
      report.writeln('## 6. Inference Timing');
      report.writeln('');
      report.writeln('| Test | VK FP32 (ms) | XN FP32 (ms) | VK FP16 (ms) |');
      report.writeln('|------|-------------|-------------|-------------|');

      for (final name in testNames) {
        final vk32 = _findResult(name, 'FP32', 'Vulkan');
        final xn32 = _findResult(name, 'FP32', 'XNNPACK');
        final vk16 = _findResult(name, 'FP16', 'Vulkan');

        report.writeln(
          '| $name | '
          '${vk32?.inferenceMs ?? "-"} | '
          '${xn32?.inferenceMs ?? "-"} | '
          '${vk16?.inferenceMs ?? "-"} |',
        );
      }

      // ------------------------------------------------------------------
      // Section 7: Conclusions
      // ------------------------------------------------------------------
      report.writeln('');
      report.writeln('## 7. Conclusions');
      report.writeln('');

      // Auto-generate conclusions
      // Check bias dropping
      bool biasDropped = false;
      for (final prefix in ['conv3x3', 'pw', 'dw']) {
        final nbVk = _findResult('${prefix}_nobias', 'FP32', 'Vulkan');
        final bVk = _findResult('${prefix}_bias', 'FP32', 'Vulkan');
        if (nbVk != null && bVk != null && nbVk.mean != null && bVk.mean != null) {
          if ((bVk.mean! - nbVk.mean!).abs() < 0.1) {
            biasDropped = true;
          }
        }
      }

      // Check FP16 broken
      bool fp16Broken = false;
      int fp16WildCount = 0;
      for (final r in allResults) {
        if (r.backend == 'Vulkan' && r.dtype == 'FP16' && r.error == null) {
          final cpuRef = cpuRefMeans['${r.name}_FP32'];
          if (cpuRef != null && r.mean != null) {
            if ((r.mean! - cpuRef).abs() > 2.0) {
              fp16Broken = true;
              fp16WildCount++;
            }
          }
          if ((r.nanCount ?? 0) > 0 || (r.infCount ?? 0) > 0) {
            fp16Broken = true;
          }
        }
      }

      // Check basic conv accuracy
      final convNoBiasVk = _findResult('conv3x3_nobias', 'FP32', 'Vulkan');
      final convNoBiasXn = _findResult('conv3x3_nobias', 'FP32', 'XNNPACK');
      bool basicConvWorks = false;
      if (convNoBiasVk != null && convNoBiasXn != null) {
        basicConvWorks =
            ((convNoBiasVk.mean ?? 0) - (convNoBiasXn.mean ?? 0)).abs() < 0.01;
      }

      // Check add_bias_only
      bool addBiasWorks = false;
      if (addBiasVk != null && addBiasVk.mean != null) {
        addBiasWorks = (addBiasVk.mean! - 1.5).abs() < 0.01;
      }

      if (basicConvWorks) {
        report.writeln(
          '1. **FP32 conv without bias: WORKS** - Basic convolution produces correct results on PowerVR',
        );
      } else {
        report.writeln(
          '1. **FP32 conv without bias: BROKEN** - Even basic convolution fails on PowerVR',
        );
      }

      if (biasDropped) {
        report.writeln(
          '2. **FP32 conv bias: DROPPED** - Bias is completely ignored in conv2d (VK output identical with/without bias)',
        );
      } else {
        report.writeln(
          '2. **FP32 conv bias: Applied correctly**',
        );
      }

      if (addBiasWorks) {
        report.writeln(
          '3. **aten.add.Tensor (add_bias_only): WORKS** - Constant tensor loading to GPU is correct',
        );
      }

      if (fp16Broken) {
        report.writeln(
          '4. **FP16: COMPLETELY BROKEN** - $fp16WildCount tests produce wildly wrong values; separate issue from bias',
        );
      } else {
        report.writeln('4. **FP16: Working correctly**');
      }

      // MobileNet analysis
      final mn1 = _findResult('mobilenet_slice_1', 'FP32', 'Vulkan');
      final mn1xn = _findResult('mobilenet_slice_1', 'FP32', 'XNNPACK');
      if (mn1 != null && mn1xn != null && mn1.mean != null && mn1xn.mean != null) {
        final diff = (mn1.mean! - mn1xn.mean!).abs();
        if (diff > 0.5) {
          report.writeln(
            '5. **MobileNet slice_1 (Conv+BN+Hardswish): DIVERGED** - First layer already fails '
            '(VK=${mn1.mean!.toStringAsFixed(4)}, XN=${mn1xn.mean!.toStringAsFixed(4)}, diff=${diff.toStringAsFixed(4)})',
          );
        }
      }

      if (biasDropped) {
        report.writeln('');
        report.writeln('### Root Cause Hypothesis');
        report.writeln('');
        report.writeln(
          'Conv2d bias (prepacked as texture2D via nchw_to_image shader) is not being '
          'read correctly on PowerVR. The bias tensor is created and prepacked, but the '
          'conv2d shader reads zeros instead of the actual bias values. Since `add_bias_only` '
          '(which uses aten.add with a buffer/texture3D constant) works, the issue is specific '
          'to how conv2d bias is stored/accessed as texture2D on PowerVR hardware.',
        );
      }

      report.writeln('');
      report.writeln('====== REPORT END ======');
      report.writeln('');

      // Print the entire report
      print(report.toString());
    });
  });
}

// =============================================================================
// Helpers
// =============================================================================

class _TestConfig {
  final String name;
  final List<int> inputShape;
  final double? expected;
  const _TestConfig(this.name, this.inputShape, this.expected);
}

TestResult? _findResult(String name, String dtype, String backend) {
  return allResults
      .where(
        (r) => r.name == name && r.dtype == dtype && r.backend == backend,
      )
      .firstOrNull;
}

String _fmtMean(TestResult? r) {
  if (r == null) return '-';
  if (r.error != null) return 'ERR';
  if (r.mean == null) return 'N/A';
  return r.mean!.toStringAsFixed(4);
}

String _getStatus(TestResult? vk, TestResult? xn) {
  if (vk == null || vk.error != null) return 'VK_FAIL';
  if ((vk.nanCount ?? 0) > 0) return 'NaN';
  if ((vk.infCount ?? 0) > 0) return 'Inf';
  if (xn == null || xn.error != null) return 'VK_ONLY';

  final diff = ((vk.mean ?? 0) - (xn.mean ?? 0)).abs();
  if (diff < 0.01) return 'MATCH';
  if (diff < 0.5) return 'CLOSE';
  return 'DIVERGED';
}

List<TensorData> _createOnesInput(List<int> shape) {
  final numel = shape.reduce((a, b) => a * b);
  final floats = Float32List(numel);
  for (int i = 0; i < numel; i++) {
    floats[i] = 1.0;
  }
  return [
    TensorData(
      shape: shape,
      dataType: TensorType.float32,
      data: floats.buffer.asUint8List(),
    ),
  ];
}

Future<void> _runAndRecord(
  String asset,
  List<TensorData> input,
  String testName,
  String dtype,
  String backend,
) async {
  ExecuTorchModel? model;
  try {
    try {
      model = await ExecuTorchModel.loadFromAsset(asset);
    } catch (e) {
      print('  [$backend $dtype] SKIP - not found: $asset');
      allResults.add(TestResult(
        name: testName,
        dtype: dtype,
        backend: backend,
        error: 'not found',
        totalElements: 0,
      ));
      return;
    }

    final sw = Stopwatch()..start();
    final outputs = await model.forward(input);
    sw.stop();

    final output = outputs.first;
    final floats = output.data.buffer.asFloat32List();

    int nanCount = 0;
    int infCount = 0;
    double sum = 0;
    double minVal = double.infinity;
    double maxVal = double.negativeInfinity;

    for (final v in floats) {
      if (v.isNaN) {
        nanCount++;
        continue;
      }
      if (v.isInfinite) {
        infCount++;
        continue;
      }
      sum += v;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    final validCount = floats.length - nanCount - infCount;
    final mean = validCount > 0 ? sum / validCount : 0.0;

    final firstN = math.min(8, floats.length);
    final first8 =
        floats.sublist(0, firstN).map((v) => v.toDouble()).toList();

    String flags = '';
    if (nanCount > 0) flags += ' NaN=$nanCount';
    if (infCount > 0) flags += ' Inf=$infCount';

    print(
      '  [$backend $dtype] ${sw.elapsedMilliseconds}ms | '
      'shape=${output.shape} | '
      'mean=${mean.toStringAsFixed(6)} | '
      'range=[${minVal.toStringAsFixed(4)}, ${maxVal.toStringAsFixed(4)}]$flags',
    );
    print(
      '  [$backend $dtype] first8: '
      '${first8.map((v) => v.toStringAsFixed(4)).toList()}',
    );

    allResults.add(TestResult(
      name: testName,
      dtype: dtype,
      backend: backend,
      mean: mean,
      min: minVal,
      max: maxVal,
      nanCount: nanCount,
      infCount: infCount,
      totalElements: floats.length,
      first8: first8,
      inferenceMs: sw.elapsedMilliseconds,
      outputShape: output.shape,
      cpuRefMean: cpuRefMeans['${testName}_$dtype'],
    ));
  } catch (e) {
    print('  [$backend $dtype] ERROR: $e');
    allResults.add(TestResult(
      name: testName,
      dtype: dtype,
      backend: backend,
      error: e.toString(),
      totalElements: 0,
    ));
  } finally {
    if (model != null) await model.dispose();
  }
}
