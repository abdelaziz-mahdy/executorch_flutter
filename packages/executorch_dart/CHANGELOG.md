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
- Scope note: asset-bundle loading and web support are deliberately not
  part of this package. They live in `executorch_flutter`, which layers
  both on top of this one. Flutter applications should keep depending on
  `executorch_flutter`.
