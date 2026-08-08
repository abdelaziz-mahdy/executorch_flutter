## 0.6.2

### Fixed

- Dependency constraints held `hooks`, `native_toolchain_cmake`, and
  `code_assets` back from their latest releases. Widened, which also recovers
  the pub.dev points lost for out-of-date dependencies.

## 0.6.1

### Fixed

- The example shown on pub.dev now leads with how to use the package —
  loading a model, running it, and disposing it — instead of notes that only
  apply inside this repository.

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
