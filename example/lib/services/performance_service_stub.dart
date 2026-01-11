/// Web stub for PerformanceService
/// Device-specific performance detection is not available on web
library;

import 'dart:async';
import 'package:flutter/foundation.dart';

/// Performance monitoring service stub for web platform
class PerformanceService {
  static final PerformanceService _instance = PerformanceService._internal();
  factory PerformanceService() => _instance;
  PerformanceService._internal();

  // Performance metrics
  final List<double> _processingTimes = [];

  // Device capabilities (defaults for web)
  late DeviceCapabilities _capabilities;

  // Performance monitoring
  Timer? _monitoringTimer;
  bool _isMonitoring = false;

  // Throttling
  bool _isThrottled = false;
  int _consecutiveSlowFrames = 0;
  static const int _maxSlowFrames = 5;
  static const double _slowFrameThreshold = 100.0; // ms

  Future<void> initialize() async {
    _capabilities = DeviceCapabilities(
      platform: 'Web',
      model: 'Browser',
      isLowEndDevice: false,
      recommendedProcessingMode: ProcessingMode.balanced,
      maxConcurrentModels: 2,
      supportsQuantization: true,
    );
    debugPrint('Device capabilities: ${_capabilities.toString()}');
  }

  void startMonitoring() {
    if (_isMonitoring) return;

    _isMonitoring = true;
    _monitoringTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      // No-op for web
    });
  }

  void stopMonitoring() {
    _isMonitoring = false;
    _monitoringTimer?.cancel();
    _monitoringTimer = null;
  }

  void recordProcessingTime(double timeMs) {
    _processingTimes.add(timeMs);

    // Keep only last 60 entries
    if (_processingTimes.length > 60) {
      _processingTimes.removeAt(0);
    }

    // Check for throttling
    if (timeMs > _slowFrameThreshold) {
      _consecutiveSlowFrames++;
      if (_consecutiveSlowFrames >= _maxSlowFrames && !_isThrottled) {
        _isThrottled = true;
        debugPrint('Performance throttling enabled');
      }
    } else {
      _consecutiveSlowFrames = 0;
      if (_isThrottled && _shouldDisableThrottling()) {
        _isThrottled = false;
        debugPrint('Performance throttling disabled');
      }
    }
  }

  bool _shouldDisableThrottling() {
    if (_processingTimes.length < 10) return false;

    final recentTimes = _processingTimes.sublist(_processingTimes.length - 10);
    final averageTime =
        recentTimes.reduce((a, b) => a + b) / recentTimes.length;

    return averageTime < _slowFrameThreshold * 0.7; // 30% buffer
  }

  // Getters for current metrics
  DeviceCapabilities get capabilities => _capabilities;
  bool get isThrottled => _isThrottled;
  bool get isMonitoring => _isMonitoring;

  double get averageProcessingTime {
    if (_processingTimes.isEmpty) return 0.0;
    return _processingTimes.reduce((a, b) => a + b) / _processingTimes.length;
  }

  double get maxProcessingTime {
    if (_processingTimes.isEmpty) return 0.0;
    return _processingTimes.reduce((a, b) => a > b ? a : b);
  }

  double get currentFPS {
    if (_processingTimes.isEmpty) return 0.0;
    final avgTime = averageProcessingTime;
    return avgTime > 0 ? 1000.0 / avgTime : 0.0;
  }

  List<double> get recentProcessingTimes => List.unmodifiable(_processingTimes);

  ProcessingRecommendation getProcessingRecommendation() {
    if (_isThrottled) {
      return ProcessingRecommendation(
        mode: ProcessingMode.efficient,
        targetFPS: 15.0,
        shouldQuantize: true,
        maxConcurrentInferences: 1,
        reason: 'Performance throttling active',
      );
    }

    final avgTime = averageProcessingTime;
    if (avgTime > 50.0) {
      return ProcessingRecommendation(
        mode: ProcessingMode.balanced,
        targetFPS: 20.0,
        shouldQuantize: true,
        maxConcurrentInferences: 1,
        reason: 'High processing time detected',
      );
    }

    return ProcessingRecommendation(
      mode: ProcessingMode.performance,
      targetFPS: 30.0,
      shouldQuantize: false,
      maxConcurrentInferences: _capabilities.maxConcurrentModels,
      reason: 'Good performance detected',
    );
  }
}

class DeviceCapabilities {
  final String platform;
  final String model;
  final bool isLowEndDevice;
  final ProcessingMode recommendedProcessingMode;
  final int maxConcurrentModels;
  final bool supportsQuantization;

  DeviceCapabilities({
    required this.platform,
    required this.model,
    required this.isLowEndDevice,
    required this.recommendedProcessingMode,
    required this.maxConcurrentModels,
    required this.supportsQuantization,
  });

  @override
  String toString() {
    return 'DeviceCapabilities(platform: $platform, model: $model, isLowEnd: $isLowEndDevice, mode: $recommendedProcessingMode)';
  }
}

class ProcessingRecommendation {
  final ProcessingMode mode;
  final double targetFPS;
  final bool shouldQuantize;
  final int maxConcurrentInferences;
  final String reason;

  ProcessingRecommendation({
    required this.mode,
    required this.targetFPS,
    required this.shouldQuantize,
    required this.maxConcurrentInferences,
    required this.reason,
  });
}

enum ProcessingMode { efficient, balanced, performance }
