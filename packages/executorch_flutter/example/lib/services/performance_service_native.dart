import 'dart:async';
import 'dart:io';

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart';

/// Performance monitoring service for ML processing optimization
class PerformanceService {
  static final PerformanceService _instance = PerformanceService._internal();
  factory PerformanceService() => _instance;
  PerformanceService._internal();

  final DeviceInfoPlugin _deviceInfo = DeviceInfoPlugin();

  // Performance metrics
  final List<double> _processingTimes = [];
  final List<double> _memoryUsage = [];
  final List<DateTime> _timestamps = [];

  // Device capabilities
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
    _capabilities = await _detectDeviceCapabilities();
    debugPrint('Device capabilities: ${_capabilities.toString()}');
  }

  Future<DeviceCapabilities> _detectDeviceCapabilities() async {
    if (Platform.isAndroid) {
      final androidInfo = await _deviceInfo.androidInfo;
      return DeviceCapabilities(
        platform: 'Android',
        model: androidInfo.model,
        isLowEndDevice: _isLowEndAndroid(androidInfo),
        recommendedProcessingMode: _getRecommendedMode(androidInfo),
        maxConcurrentModels: _isLowEndAndroid(androidInfo) ? 1 : 2,
        supportsQuantization: true,
      );
    } else if (Platform.isIOS) {
      final iosInfo = await _deviceInfo.iosInfo;
      return DeviceCapabilities(
        platform: 'iOS',
        model: iosInfo.model,
        isLowEndDevice: _isLowEndIOS(iosInfo),
        recommendedProcessingMode: _getRecommendedModeiOS(iosInfo),
        maxConcurrentModels: _isLowEndIOS(iosInfo) ? 1 : 3,
        supportsQuantization: true,
      );
    }

    return DeviceCapabilities(
      platform: 'Unknown',
      model: 'Unknown',
      isLowEndDevice: true,
      recommendedProcessingMode: ProcessingMode.efficient,
      maxConcurrentModels: 1,
      supportsQuantization: false,
    );
  }

  bool _isLowEndAndroid(AndroidDeviceInfo info) {
    // Simple heuristic based on Android version and available memory
    return info.version.sdkInt < 26; // Android 8.0+
  }

  bool _isLowEndIOS(IosDeviceInfo info) {
    // Simple heuristic based on device model
    final model = info.model.toLowerCase();
    return model.contains('iphone 6') ||
        model.contains('iphone 7') ||
        model.contains('iphone se');
  }

  ProcessingMode _getRecommendedMode(AndroidDeviceInfo info) {
    if (info.version.sdkInt >= 29) {
      return ProcessingMode.performance;
    } else if (info.version.sdkInt >= 26) {
      return ProcessingMode.balanced;
    }
    return ProcessingMode.efficient;
  }

  ProcessingMode _getRecommendedModeiOS(IosDeviceInfo info) {
    final model = info.model.toLowerCase();
    if (model.contains('iphone 13') ||
        model.contains('iphone 14') ||
        model.contains('iphone 15')) {
      return ProcessingMode.performance;
    } else if (model.contains('iphone 11') || model.contains('iphone 12')) {
      return ProcessingMode.balanced;
    }
    return ProcessingMode.efficient;
  }

  void startMonitoring() {
    if (_isMonitoring) return;

    _isMonitoring = true;
    _monitoringTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _collectMetrics();
    });
  }

  void stopMonitoring() {
    _isMonitoring = false;
    _monitoringTimer?.cancel();
    _monitoringTimer = null;
  }

  void _collectMetrics() {
    final now = DateTime.now();
    _timestamps.add(now);

    // Keep only last 60 seconds of data
    final cutoff = now.subtract(const Duration(seconds: 60));
    while (_timestamps.isNotEmpty && _timestamps.first.isBefore(cutoff)) {
      _timestamps.removeAt(0);
      if (_processingTimes.isNotEmpty) _processingTimes.removeAt(0);
      if (_memoryUsage.isNotEmpty) _memoryUsage.removeAt(0);
    }
  }

  void recordProcessingTime(double timeMs) {
    _processingTimes.add(timeMs);

    // Check for throttling
    if (timeMs > _slowFrameThreshold) {
      _consecutiveSlowFrames++;
      if (_consecutiveSlowFrames >= _maxSlowFrames && !_isThrottled) {
        _enableThrottling();
      }
    } else {
      _consecutiveSlowFrames = 0;
      if (_isThrottled && _shouldDisableThrottling()) {
        _disableThrottling();
      }
    }
  }

  void _enableThrottling() {
    _isThrottled = true;
    debugPrint('Performance throttling enabled');
  }

  void _disableThrottling() {
    _isThrottled = false;
    _consecutiveSlowFrames = 0;
    debugPrint('Performance throttling disabled');
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
    if (_capabilities.isLowEndDevice || _isThrottled) {
      return ProcessingRecommendation(
        mode: ProcessingMode.efficient,
        targetFPS: 15.0,
        shouldQuantize: true,
        maxConcurrentInferences: 1,
        reason: _isThrottled
            ? 'Performance throttling active'
            : 'Low-end device detected',
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
