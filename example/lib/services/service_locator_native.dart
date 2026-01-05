/// Native platform service locator implementation
library;

import 'dart:io';
import 'package:get_it/get_it.dart';
import '../controllers/camera_controller.dart';
import '../controllers/opencv_camera_controller.dart';
import '../controllers/platform_camera_controller.dart';
import '../processors/camera_image_converter.dart';
import 'processor_preferences.dart';

final _getIt = GetIt.instance;

/// Initialize all services and controllers for native platforms
Future<void> setupServiceLocatorNative() async {
  // Register camera controller as lazy singleton
  _getIt.registerLazySingleton<CameraController>(() {
    if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
      // Desktop: Use OpenCV camera
      return OpenCVCameraController(
        deviceId: 0,
        processingInterval: const Duration(milliseconds: 100),
      );
    } else {
      // Mobile: Use platform camera
      return PlatformCameraController(
        converter: ImageLibCameraConverter(),
        processingInterval: const Duration(milliseconds: 100),
      );
    }
  });
}

/// Update camera controller based on processor preference (mobile only)
Future<void> updateCameraConverterNative() async {
  if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
    return; // Desktop uses OpenCV, no converter to update
  }

  final useOpenCV = await ProcessorPreferences.getUseOpenCV();
  final CameraImageConverter converter =
      useOpenCV ? OpenCVCameraConverter() : ImageLibCameraConverter();

  // Recreate platform camera controller with new converter
  if (_getIt.isRegistered<CameraController>()) {
    final oldController = _getIt<CameraController>();
    await oldController.dispose();
    await _getIt.unregister<CameraController>();
  }

  _getIt.registerLazySingleton<CameraController>(() {
    return PlatformCameraController(
      converter: converter,
      processingInterval: const Duration(milliseconds: 100),
    );
  });
}
