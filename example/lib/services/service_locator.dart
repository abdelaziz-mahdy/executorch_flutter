import 'dart:io';
import 'package:get_it/get_it.dart';
import '../controllers/camera_controller.dart';
import '../controllers/opencv_camera_controller.dart';
import '../controllers/platform_camera_controller.dart';
import '../processors/camera_image_converter.dart';
import 'processor_preferences.dart';

final getIt = GetIt.instance;

/// Initialize all services and controllers
Future<void> setupServiceLocator() async {
  // Register camera controller as lazy singleton
  // It will be created when first accessed and persists for app lifetime
  getIt.registerLazySingleton<CameraController>(() {
    if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
      // Desktop: Use OpenCV camera
      return OpenCVCameraController(
        deviceId: 0,
        processingInterval: const Duration(milliseconds: 100),
      );
    } else {
      // Mobile: Use platform camera (will be updated based on processor preference)
      return PlatformCameraController(
        converter: ImageLibCameraConverter(), // Default converter
        processingInterval: const Duration(milliseconds: 100),
      );
    }
  });
}

/// Update camera controller based on processor preference (mobile only)
Future<void> updateCameraConverter() async {
  if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
    return; // Desktop uses OpenCV, no converter to update
  }

  final useOpenCV = await ProcessorPreferences.getUseOpenCV();
  final converter = useOpenCV
      ? OpenCVCameraConverter()
      : ImageLibCameraConverter();

  // Recreate platform camera controller with new converter
  if (getIt.isRegistered<CameraController>()) {
    final oldController = getIt<CameraController>();
    await oldController.dispose();
    await getIt.unregister<CameraController>();
  }

  getIt.registerLazySingleton<CameraController>(() {
    return PlatformCameraController(
      converter: converter,
      processingInterval: const Duration(milliseconds: 100),
    );
  });
}
