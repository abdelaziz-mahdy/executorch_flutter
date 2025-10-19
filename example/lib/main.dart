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
import 'screens/unified_model_playground.dart';

// Import services
import 'services/performance_service.dart';
import 'services/service_locator.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize service locator (camera controllers, etc.)
  try {
    await setupServiceLocator();
    debugPrint('✅ Service Locator initialized successfully');
  } catch (e) {
    debugPrint('❌ Failed to initialize Service Locator: $e');
  }

  // ExecuTorch FFI bridge initializes automatically on first model load
  // No explicit initialization needed
  debugPrint('✅ ExecuTorch FFI ready (initializes on first use)');

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
