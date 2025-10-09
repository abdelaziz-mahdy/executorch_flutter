import 'dart:io';
import 'dart:typed_data';

/// Base class for all model input types
/// This allows the architecture to support multiple input modes without breaking changes
abstract class ModelInput {}

/// Input for static image files (e.g., gallery selection)
class ImageFileInput extends ModelInput {
  final File file;

  ImageFileInput(this.file);
}

/// Input for live camera frames
/// Only holds the raw frame bytes for rendering (processing happens separately)
class LiveCameraInput extends ModelInput {
  final Uint8List frameBytes; // Raw frame bytes for Image.memory rendering

  LiveCameraInput(this.frameBytes);
}
