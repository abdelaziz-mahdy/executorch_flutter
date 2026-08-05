import 'dart:typed_data';

/// Base class for all model input types
/// This allows the architecture to support multiple input modes without breaking changes
abstract class ModelInput {}

/// Input for images from bytes (works on all platforms including web)
/// This is the recommended input type for cross-platform compatibility.
class ImageBytesInput extends ModelInput {
  final Uint8List imageBytes;

  ImageBytesInput(this.imageBytes);
}

/// Input for live camera frames
/// Only holds the raw frame bytes for rendering (processing happens separately)
class LiveCameraInput extends ModelInput {
  final Uint8List frameBytes; // Raw frame bytes for Image.memory rendering

  LiveCameraInput(this.frameBytes);
}

/// Input for text generation models (e.g., Gemma, GPT, etc.)
class TextPromptInput extends ModelInput {
  final String text;

  TextPromptInput(this.text);
}
