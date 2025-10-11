import 'package:flutter/material.dart';

/// Performance data class for tracking inference metrics
class PerformanceMetrics {
  // Average times (for camera mode)
  final double? preprocessingTime;
  final double? inferenceTime;
  final double? postprocessingTime;
  final double? totalTime;

  // Current frame times (for camera mode)
  final double? currentPreprocessingTime;
  final double? currentInferenceTime;
  final double? currentPostprocessingTime;
  final double? currentTotalTime;

  final int? frameCount; // For camera mode
  final double? fps; // For camera mode

  const PerformanceMetrics({
    this.preprocessingTime,
    this.inferenceTime,
    this.postprocessingTime,
    this.totalTime,
    this.currentPreprocessingTime,
    this.currentInferenceTime,
    this.currentPostprocessingTime,
    this.currentTotalTime,
    this.frameCount,
    this.fps,
  });

  bool get hasData =>
      preprocessingTime != null &&
      inferenceTime != null &&
      postprocessingTime != null &&
      totalTime != null;

  bool get isCameraMode => frameCount != null;
}

/// Display mode for performance monitor
enum PerformanceDisplayMode {
  overlay, // Floating overlay (for camera)
  section, // Inline section (for static images)
}

/// Generic performance monitor widget
/// Displays timing metrics in either overlay or section mode
class PerformanceMonitor extends StatelessWidget {
  final PerformanceMetrics metrics;
  final PerformanceDisplayMode displayMode;

  const PerformanceMonitor({
    super.key,
    required this.metrics,
    this.displayMode = PerformanceDisplayMode.section,
  });

  @override
  Widget build(BuildContext context) {
    if (!metrics.hasData) return const SizedBox.shrink();

    return displayMode == PerformanceDisplayMode.overlay
        ? _buildOverlay(context)
        : _buildSection(context);
  }

  Widget _buildOverlay(BuildContext context) {
    final fps = metrics.fps ?? 0.0;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.2),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.speed,
                size: 16,
                color: Colors.white.withValues(alpha: 0.9),
              ),
              const SizedBox(width: 6),
              Text(
                'Live Performance',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.9),
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),

          // Frames processed (camera mode only)
          if (metrics.frameCount != null) ...[
            _buildMetricRow('Frames', '${metrics.frameCount}', Colors.blue),
            const SizedBox(height: 4),
          ],

          // FPS (camera mode only)
          if (metrics.fps != null) ...[
            _buildMetricRow(
              'FPS',
              fps.toStringAsFixed(1),
              fps > 20 ? Colors.green : (fps > 10 ? Colors.orange : Colors.red),
            ),
            const SizedBox(height: 8),
          ],

          // Current frame metrics (camera mode only)
          if (metrics.isCameraMode && metrics.currentTotalTime != null) ...[
            Container(height: 1, color: Colors.white.withValues(alpha: 0.2)),
            const SizedBox(height: 8),
            Text(
              'Current Frame',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 6),
            _buildMetricRow(
              'Preprocess',
              '${metrics.currentPreprocessingTime!.toStringAsFixed(1)}ms',
              Colors.blue,
            ),
            const SizedBox(height: 4),
            _buildMetricRow(
              'Inference',
              '${metrics.currentInferenceTime!.toStringAsFixed(1)}ms',
              Colors.green,
            ),
            const SizedBox(height: 4),
            _buildMetricRow(
              'Postprocess',
              '${metrics.currentPostprocessingTime!.toStringAsFixed(1)}ms',
              Colors.orange,
            ),
            const SizedBox(height: 4),
            _buildMetricRow(
              'Total',
              '${metrics.currentTotalTime!.toStringAsFixed(1)}ms',
              Colors.white,
              isBold: true,
            ),
            const SizedBox(height: 8),
          ],

          // Divider
          if (metrics.isCameraMode) ...[
            Container(height: 1, color: Colors.white.withValues(alpha: 0.2)),
            const SizedBox(height: 8),
            Text(
              'Average',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 6),
          ],

          // Timing metrics (average for camera, single value for static)
          _buildMetricRow(
            'Preprocess',
            '${metrics.preprocessingTime!.toStringAsFixed(1)}ms',
            Colors.blue,
          ),
          const SizedBox(height: 4),

          _buildMetricRow(
            'Inference',
            '${metrics.inferenceTime!.toStringAsFixed(1)}ms',
            Colors.green,
          ),
          const SizedBox(height: 4),

          _buildMetricRow(
            'Postprocess',
            '${metrics.postprocessingTime!.toStringAsFixed(1)}ms',
            Colors.orange,
          ),
          const SizedBox(height: 4),

          _buildMetricRow(
            'Total',
            '${metrics.totalTime!.toStringAsFixed(1)}ms',
            Colors.white,
            isBold: true,
          ),
        ],
      ),
    );
  }

  Widget _buildSection(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border(
          top: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
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
                'Performance',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
              ),
              const Spacer(),
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
                  '${metrics.totalTime!.toStringAsFixed(0)}ms',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).colorScheme.onPrimaryContainer,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Timing breakdown with progress bars
          _buildTimingRow(
            context,
            'Preprocessing',
            metrics.preprocessingTime!,
            metrics.totalTime!,
            Colors.blue,
          ),
          const SizedBox(height: 8),

          _buildTimingRow(
            context,
            'Inference',
            metrics.inferenceTime!,
            metrics.totalTime!,
            Colors.green,
          ),
          const SizedBox(height: 8),

          _buildTimingRow(
            context,
            'Postprocessing',
            metrics.postprocessingTime!,
            metrics.totalTime!,
            Colors.orange,
          ),
        ],
      ),
    );
  }

  Widget _buildMetricRow(
    String label,
    String value,
    Color color, {
    bool isBold = false,
  }) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.8),
            fontSize: 11,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 11,
            fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ],
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
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(label, style: Theme.of(context).textTheme.bodyMedium),
            ),
            Text(
              '${time.toStringAsFixed(0)}ms',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.bold),
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
            backgroundColor: Theme.of(
              context,
            ).colorScheme.surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}

/// Helper class for tracking running performance averages (camera mode)
class PerformanceTracker {
  int _frameCount = 0;
  double _avgPreprocessingTime = 0.0;
  double _avgInferenceTime = 0.0;
  double _avgPostprocessingTime = 0.0;
  double _avgTotalTime = 0.0;

  // Current frame metrics
  double _currentPreprocessingTime = 0.0;
  double _currentInferenceTime = 0.0;
  double _currentPostprocessingTime = 0.0;
  double _currentTotalTime = 0.0;

  int get frameCount => _frameCount;
  double get fps => _avgTotalTime > 0 ? (1000 / _avgTotalTime) : 0.0;

  void reset() {
    _frameCount = 0;
    _avgPreprocessingTime = 0.0;
    _avgInferenceTime = 0.0;
    _avgPostprocessingTime = 0.0;
    _avgTotalTime = 0.0;
    _currentPreprocessingTime = 0.0;
    _currentInferenceTime = 0.0;
    _currentPostprocessingTime = 0.0;
    _currentTotalTime = 0.0;
  }

  void update({
    required double preprocessingTime,
    required double inferenceTime,
    required double postprocessingTime,
    required double totalTime,
  }) {
    _frameCount++;

    // Update current frame metrics
    _currentPreprocessingTime = preprocessingTime;
    _currentInferenceTime = inferenceTime;
    _currentPostprocessingTime = postprocessingTime;
    _currentTotalTime = totalTime;

    // Update running averages
    _avgPreprocessingTime =
        (_avgPreprocessingTime * (_frameCount - 1) + preprocessingTime) /
        _frameCount;
    _avgInferenceTime =
        (_avgInferenceTime * (_frameCount - 1) + inferenceTime) / _frameCount;
    _avgPostprocessingTime =
        (_avgPostprocessingTime * (_frameCount - 1) + postprocessingTime) /
        _frameCount;
    _avgTotalTime =
        (_avgTotalTime * (_frameCount - 1) + totalTime) / _frameCount;
  }

  PerformanceMetrics toMetrics() {
    return PerformanceMetrics(
      preprocessingTime: _avgPreprocessingTime,
      inferenceTime: _avgInferenceTime,
      postprocessingTime: _avgPostprocessingTime,
      totalTime: _avgTotalTime,
      currentPreprocessingTime: _currentPreprocessingTime,
      currentInferenceTime: _currentInferenceTime,
      currentPostprocessingTime: _currentPostprocessingTime,
      currentTotalTime: _currentTotalTime,
      frameCount: _frameCount,
      fps: fps,
    );
  }
}
