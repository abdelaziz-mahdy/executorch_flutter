/// ExecuTorch Flutter Example App
///
/// A modern model playground showcasing ExecuTorch integration with Flutter.
///
/// Features:
/// - Single-page adaptive UI
/// - Image Classification with MobileNet
/// - Object Detection with YOLO
/// - Real-time camera processing
/// - Modern Material 3 design
library;

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:image_picker_platform_interface/image_picker_platform_interface.dart';

// Import model playground
import 'screens/unified_model_playground.dart';

// Import services
import 'services/performance_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Set up camera delegate for desktop platforms (macOS, Windows, Linux)
  if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
    final ImagePickerPlatform instance = ImagePickerPlatform.instance;
    if (instance is CameraDelegatingImagePickerPlatform) {
      // For now, camera is not supported on desktop - use gallery only
      // Users should use ImageSource.gallery instead of ImageSource.camera
      debugPrint('⚠️  Camera not supported on desktop platforms. Use gallery instead.');
    }
  }

  // Initialize ExecuTorch manager
  try {
    await ExecutorchManager.instance.initialize();
    debugPrint('✅ ExecuTorch Manager initialized successfully');

    // Enable debug logging to see detailed ExecuTorch logs
    await ExecutorchManager.instance.setDebugLogging(true);
    debugPrint('✅ ExecuTorch debug logging enabled');
  } catch (e) {
    debugPrint('❌ Failed to initialize ExecuTorch: $e');
  }

  // Initialize Performance Service
  try {
    await PerformanceService().initialize();
    debugPrint('✅ Performance Service initialized successfully');
  } catch (e) {
    debugPrint('❌ Failed to initialize Performance Service: $e');
  }

  runApp(const ExecuTorchPlaygroundApp());
}

class ExecuTorchPlaygroundApp extends StatelessWidget {
  const ExecuTorchPlaygroundApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ExecuTorch Playground',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6750A4),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6750A4),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const UnifiedModelPlayground(),
      debugShowCheckedModeBanner: false,
    );
  }
}