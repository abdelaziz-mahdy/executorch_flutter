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

import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

// Import model playground
import 'screens/model_playground.dart';

// Import services
import 'services/performance_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize ExecuTorch manager
  try {
    await ExecutorchManager.instance.initialize();
    debugPrint('✅ ExecuTorch Manager initialized successfully');
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
      home: const ModelPlayground(),
      debugShowCheckedModeBanner: false,
    );
  }
}