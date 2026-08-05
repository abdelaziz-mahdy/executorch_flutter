/// Web platform helpers for unified model playground
library;

/// Load a model asset to a file path - NOT supported on web
///
/// On web, models should be loaded directly from bytes using
/// ExecuTorchModel.loadFromBytes() instead.
Future<String> loadAssetToFile(String assetPath) async {
  throw UnsupportedError(
    'loadAssetToFile is not supported on web. '
    'Use ExecuTorchModel.loadFromBytes() instead.',
  );
}
