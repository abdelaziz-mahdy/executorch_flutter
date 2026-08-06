# Changelog

## 0.6.0

### Added

- Initial release of `executorch_dart`, the pure-Dart core extracted from
  `executorch_flutter`. It owns the FFI layer, the native-assets build hook,
  and the native library, so ExecuTorch inference now runs in any Dart
  program — servers and command-line tools included — with no Flutter SDK.
- Vision inference through `ExecuTorchModel` (`load`, `loadFromBytes`,
  `forward`, `dispose`), the `ExecutorchManager` facade, experimental
  streaming LLM through `ExecuTorchLLM`, backend and version queries, and
  the base processors.

### Breaking

- Asset-bundle loading is not part of this package. `loadFromAsset` and
  `loadModelFromAssets` live in `executorch_flutter`, which layers asset
  loading and web support on top of this package. Flutter applications
  should keep depending on `executorch_flutter`.
