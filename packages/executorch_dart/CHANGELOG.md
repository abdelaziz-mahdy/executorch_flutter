## 0.6.0

### Added

- First release. Pure-Dart ExecuTorch inference over dart:ffi, extracted from
  `executorch_flutter` so Dart servers and command-line programs can run models
  without a Flutter SDK. Owns the native build hook and the prebuilt binary
  download.
- `executorch_dart_shared.dart` (the ffi-free half of the public API) and the
  `ExecutorchManagerBase` class are now part of the public API surface. Both
  exist so `executorch_flutter`'s web implementation can build on this
  package without reaching into its private internals, and are expected to
  stay stable like the rest of the public API.
