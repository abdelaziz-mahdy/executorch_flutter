/// Web platform stub for test images
library;

/// Get a temporary file from an asset image
/// Not supported on web - throws UnsupportedError
Future<Never> getFileFromAsset(String assetPath) async {
  throw UnsupportedError(
    'TestImages.getFileFromAsset() is not supported on web. '
    'Use TestImages.getBytesFromAsset() instead.',
  );
}
