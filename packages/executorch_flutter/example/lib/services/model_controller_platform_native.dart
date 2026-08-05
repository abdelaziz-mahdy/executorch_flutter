/// Native platform helpers for model controller
library;

import 'dart:io';
import '../models/model_settings.dart';

/// Check if current platform is desktop
bool get isDesktopPlatform =>
    Platform.isMacOS || Platform.isWindows || Platform.isLinux;

/// Get default camera provider for native platforms
CameraProvider getDefaultCameraProvider() {
  if (isDesktopPlatform) {
    return CameraProvider.opencv;
  }
  return CameraProvider.platform;
}
