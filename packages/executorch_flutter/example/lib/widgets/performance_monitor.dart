import 'package:flutter/material.dart';
import '../services/performance_service.dart';

class PerformanceMonitor extends StatefulWidget {
  const PerformanceMonitor({super.key});

  @override
  State<PerformanceMonitor> createState() => _PerformanceMonitorState();
}

class _PerformanceMonitorState extends State<PerformanceMonitor> {
  final PerformanceService _performanceService = PerformanceService();
  bool _isExpanded = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      margin: const EdgeInsets.all(8),
      child: Column(
        children: [
          ListTile(
            leading: Icon(
              Icons.speed,
              color: _performanceService.isThrottled
                  ? Colors.orange
                  : Colors.green,
            ),
            title: const Text('Performance Monitor'),
            subtitle: Text(_getStatusText()),
            trailing: IconButton(
              icon: Icon(_isExpanded ? Icons.expand_less : Icons.expand_more),
              onPressed: () {
                setState(() {
                  _isExpanded = !_isExpanded;
                });
              },
            ),
          ),
          if (_isExpanded) ...[
            const Divider(),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildMetricsRow(theme),
                  const SizedBox(height: 16),
                  _buildDeviceInfo(theme),
                  const SizedBox(height: 16),
                  _buildRecommendations(theme),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  String _getStatusText() {
    if (_performanceService.isThrottled) {
      return 'Throttled - Performance optimization active';
    } else if (_performanceService.isMonitoring) {
      return 'Monitoring - ${_performanceService.currentFPS.toStringAsFixed(1)} FPS';
    } else {
      return 'Ready to monitor performance';
    }
  }

  Widget _buildMetricsRow(ThemeData theme) {
    return Row(
      children: [
        Expanded(
          child: _buildMetricCard(
            'Avg Time',
            '${_performanceService.averageProcessingTime.toStringAsFixed(1)}ms',
            Icons.timer,
            theme.colorScheme.primary,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _buildMetricCard(
            'Max Time',
            '${_performanceService.maxProcessingTime.toStringAsFixed(1)}ms',
            Icons.trending_up,
            Colors.orange,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _buildMetricCard(
            'FPS',
            _performanceService.currentFPS.toStringAsFixed(1),
            Icons.refresh,
            Colors.green,
          ),
        ),
      ],
    );
  }

  Widget _buildMetricCard(
    String label,
    String value,
    IconData icon,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 16, color: color),
              const SizedBox(width: 4),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  color: color,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildDeviceInfo(ThemeData theme) {
    final capabilities = _performanceService.capabilities;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.phone_android,
                size: 16,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(width: 8),
              Text(
                'Device Information',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _buildInfoRow('Platform', capabilities.platform),
          _buildInfoRow('Model', capabilities.model),
          _buildInfoRow(
            'Performance Class',
            capabilities.isLowEndDevice ? 'Basic' : 'High-end',
          ),
          _buildInfoRow('Max Models', '${capabilities.maxConcurrentModels}'),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 12)),
          Text(
            value,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendations(ThemeData theme) {
    final recommendation = _performanceService.getProcessingRecommendation();

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.primaryContainer.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb, size: 16, color: theme.colorScheme.primary),
              const SizedBox(width: 8),
              Text(
                'Recommendations',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _buildRecommendationRow('Mode', _getModeString(recommendation.mode)),
          _buildRecommendationRow(
            'Target FPS',
            recommendation.targetFPS.toStringAsFixed(0),
          ),
          _buildRecommendationRow(
            'Quantization',
            recommendation.shouldQuantize ? 'Enabled' : 'Disabled',
          ),
          _buildRecommendationRow(
            'Max Concurrent',
            '${recommendation.maxConcurrentInferences}',
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: theme.colorScheme.primaryContainer,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              recommendation.reason,
              style: TextStyle(
                fontSize: 11,
                color: theme.colorScheme.onPrimaryContainer,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendationRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 12)),
          Text(
            value,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  String _getModeString(ProcessingMode mode) {
    switch (mode) {
      case ProcessingMode.efficient:
        return 'Efficient';
      case ProcessingMode.balanced:
        return 'Balanced';
      case ProcessingMode.performance:
        return 'Performance';
    }
  }
}
