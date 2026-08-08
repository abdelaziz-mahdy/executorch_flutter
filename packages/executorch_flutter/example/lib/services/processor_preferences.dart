import 'package:shared_preferences/shared_preferences.dart';

/// Service for managing processor selection preferences
class ProcessorPreferences {
  ProcessorPreferences._();

  static const _useOpenCVKey = 'use_opencv_processor';

  /// Get whether to use OpenCV processor (true) or pure Dart (false)
  static Future<bool> getUseOpenCV() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_useOpenCVKey) ?? false; // Default to Dart processor
  }

  /// Set whether to use OpenCV processor
  static Future<void> setUseOpenCV(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_useOpenCVKey, value);
  }
}
