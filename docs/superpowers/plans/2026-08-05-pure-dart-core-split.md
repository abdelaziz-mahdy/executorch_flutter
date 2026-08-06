# Pure-Dart Core Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `executorch_flutter` into a pure-Dart core (`executorch_dart`) that owns the FFI layer and native build, plus a thin Flutter wrapper that adds asset loading and web, so non-Flutter Dart programs can run ExecuTorch inference.

**Architecture:** A pub workspace at the repo root holds two published packages under `packages/`. The core owns `hook/`, the `native/` submodule, and every `dart:ffi` path. The wrapper re-exports the core, routes web through conditional exports whose native branch points back at the core, and adds `rootBundle` helpers. Two breaking changes are guarded by mechanism: a no-asset tripwire hook that fails the build on the old `user_defines` key, and a `fix_data.yaml` that makes `dart fix --apply` rewrite the removed static.

**Tech Stack:** Dart 3.12.2, Flutter 3.38+, pub workspaces, `hooks`/`code_assets` native assets, ffigen 20.x, CMake.

**Design spec:** `docs/superpowers/specs/2026-08-05-pure-dart-core-split-design.md`

## Global Constraints

- Dart SDK constraint stays `'>=3.6.0 <4.0.0'`; Flutter constraint stays `'>=3.27.0'`.
- Every workspace member pubspec must declare `resolution: workspace`. Pub workspaces fail without it.
- Both packages ship version `0.6.0`. The wrapper declares `executorch_dart: ^0.6.0`.
- `executorchVersion` stays `'1.3.1'` (`lib/src/version.dart`). `_defaultPrebuiltVersion` stays `'$executorchVersion.9'` — do not change it. (`CLAUDE.md` documents this as `.1`; `CLAUDE.md` is stale and Task 8 Step 5 fixes it.)
- Legacy `user_defines` keys the tripwire must detect, exactly these seven: `build_mode`, `backends`, `llm`, `debug`, `local_lib_dir`, `executorch_source`, `prebuilt_version`.
- `analysis_options.yaml` enables `public_member_api_docs` and `lines_longer_than_80_chars`. Every new public member needs a `///` doc comment and every line must be 80 characters or fewer.
- `implementation_imports` is enabled. A package must never import another package's `src/`. Anything the wrapper needs from the core must be exported from the core's public library.
- Never rewrite git history: no `git commit --amend`, no `git push --force`, no `git push --force-with-lease`. Add a new commit instead.
- Never create GitHub releases or tags by hand. CI owns releases.
- Commit messages use conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `refactor:`, `test:`). Never mention AI tools and never add AI co-author trailers.
- Work on branch `feat/pure-dart-core-split`, which already exists and holds the design spec.

---

## File Structure

**Repo root after the split:**

| Path | Responsibility |
|---|---|
| `pubspec.yaml` | Workspace root only. `publish_to: none`, lists all four members. |
| `models/` | Submodule, unchanged, stays at root. |
| `docs/`, `scripts/`, `Dockerfile.wasm`, `CONTRIBUTING.md`, `ROADMAP.md` | Repo-level, stay at root. |

**`packages/executorch_dart/` — published, pure Dart, native only:**

| Path | Responsibility |
|---|---|
| `lib/executorch_dart.dart` | Public library. No conditional web branches. |
| `lib/src/executorch_model.dart` | `ExecuTorchModel` interface plus `load`/`loadFromBytes` statics. No asset static. |
| `lib/src/executorch_inference.dart` | `ExecutorchManager` interface. No `loadModelFromAssets`. |
| `lib/src/executorch_manager_base.dart` | `ExecutorchManagerBase`, now publicly exported so the wrapper's web manager can extend it. |
| `lib/src/ffi/`, `lib/src/generated/` | FFI layer and bindings, asset id `package:executorch_dart/executorch_dart.dart`. |
| `lib/src/processors/`, `lib/src/types.dart`, `lib/src/executorch_errors.dart`, `lib/src/version.dart` | Unchanged, moved. |
| `hook/build.dart`, `lib/src/build/run_build.dart` | The real native build. |
| `native/` | Submodule. Must live inside this package: the hook resolves `packageRoot/native/`. |
| `example/` | Pure-Dart CLI proving no-Flutter inference. |

**`packages/executorch_flutter/` — published, Flutter wrapper:**

| Path | Responsibility |
|---|---|
| `lib/executorch_flutter.dart` | Blanket re-export with `hide`, plus one conditional route per hidden name. |
| `lib/src/assets.dart` | `loadModelFromAsset` and the `ExecutorchManagerAssets` extension. |
| `lib/src/web/` | Web implementations, class names matching the core names they route. |
| `lib/fix_data.yaml` | `dart fix` migration for the removed static. |
| `hook/build.dart` | Tripwire only. Emits no assets. |
| `web/`, `bin/setup_web.dart` | Wasm payload and the setup executable. |
| `android/ ios/ macos/ linux/ windows/` | Plugin platform directories. |
| `example/` | Today's Flutter example app and its integration tests. |

---

### Task 1: Workspace scaffold and package move

Move the existing package under `packages/` with zero code edits, so the tree is green before anything is renamed.

**Files:**
- Create: `pubspec.yaml` (new workspace root, replacing the package one)
- Create: `README.md` (new root readme)
- Move: everything listed in Step 3 into `packages/executorch_flutter/`
- Modify: `.gitmodules`
- Modify: `packages/executorch_flutter/pubspec.yaml`, `packages/executorch_flutter/example/pubspec.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: a pub workspace whose single member is `executorch_flutter` at `packages/executorch_flutter`, with its example at `packages/executorch_flutter/example`. Package name, public API, and every import path are unchanged.

- [ ] **Step 1: Restore the deleted `models` submodule**

The working tree currently shows `D models`. Submodule surgery on a dirty tree loses work.

```bash
git submodule update --init --recursive
git status --short
```

Expected: no `D models` line. If `example/macos/Podfile.lock`, `example/macos/Runner.xcodeproj/project.pbxproj`, or `example/pubspec.lock` still show as modified, leave them — they are unrelated and not ours to commit.

- [ ] **Step 2: Record the baseline so you can prove nothing regressed**

```bash
flutter pub get
flutter analyze lib 2>&1 | tail -3
flutter test 2>&1 | tail -3
```

Expected: analyze reports `No issues found!` and tests pass. Write the numbers down; Step 8 must match them.

- [ ] **Step 3: Move the package**

```bash
mkdir -p packages/executorch_flutter
git mv android ios macos linux windows \
       lib hook web bin test native example \
       ffigen.yaml ffigen_llm.yaml analysis_options.yaml \
       pubspec.yaml CHANGELOG.md README.md .pubignore .metadata \
       packages/executorch_flutter/
cp LICENSE packages/executorch_flutter/LICENSE
```

`git mv` on the `native` submodule rewrites its `.gitmodules` path automatically in git 2.50. Verify next step.

- [ ] **Step 4: Verify the submodule path was rewritten**

```bash
grep -A2 'submodule "native"' .gitmodules
```

Expected: `path = packages/executorch_flutter/native`. If it still says `path = native`, edit `.gitmodules` by hand to match, then run `git submodule sync`.

- [ ] **Step 5: Write the workspace root pubspec**

```yaml
name: executorch_workspace
description: >
  Workspace root for the executorch_dart and executorch_flutter packages.
  Not published.
publish_to: none

environment:
  sdk: '>=3.6.0 <4.0.0'

workspace:
  - packages/executorch_flutter
  - packages/executorch_flutter/example
```

- [ ] **Step 6: Add `resolution: workspace` to both member pubspecs**

In `packages/executorch_flutter/pubspec.yaml`, directly under the `environment:` block:

```yaml
environment:
  sdk: '>=3.6.0 <4.0.0'
  flutter: '>=3.27.0'

resolution: workspace
```

Do the same in `packages/executorch_flutter/example/pubspec.yaml`:

```yaml
environment:
  sdk: ^3.9.2

resolution: workspace
```

- [ ] **Step 7: Write a minimal root README**

```markdown
# executorch_flutter

ExecuTorch on-device ML inference for Dart and Flutter.

| Package | Description |
|---|---|
| [`packages/executorch_flutter`](packages/executorch_flutter) | Flutter plugin: asset loading, web, all platforms. |

See [`packages/executorch_flutter/README.md`](packages/executorch_flutter/README.md)
for usage.
```

- [ ] **Step 8: Verify the move changed nothing**

```bash
flutter pub get
flutter analyze lib 2>&1 | tail -3
flutter test 2>&1 | tail -3
```

Expected: identical to the Step 2 baseline. The example's `path: ../` dependency still resolves, because the example moved with its package.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: move package into packages/ pub workspace

No code changes. Prepares the tree for extracting a pure-Dart core."
```

---

### Task 2: Extract the `executorch_dart` core

Create the pure-Dart package and move every non-Flutter file into it. At the end of this task the core analyzes and tests on its own with no Flutter in its dependency graph. The wrapper is deliberately left broken; Task 3 rebuilds it.

**Files:**
- Create: `packages/executorch_dart/pubspec.yaml`, `lib/executorch_dart.dart`, `analysis_options.yaml`, `CHANGELOG.md`, `.pubignore`, `LICENSE`
- Move: core sources, `hook/`, `native/`, `test/`, ffigen configs, out of `packages/executorch_flutter/`
- Modify: `lib/src/build/run_build.dart:71,428`, both generated binding files, both ffigen configs, `lib/src/executorch_model.dart`, `lib/src/executorch_inference.dart`, `lib/src/executorch_manager_base.dart`, the three model stubs, the three manager stubs
- Modify: `.gitmodules`, root `pubspec.yaml`

**Interfaces:**
- Consumes: the workspace from Task 1.
- Produces:
  - Package `executorch_dart`, library `package:executorch_dart/executorch_dart.dart`.
  - Native asset id `package:executorch_dart/executorch_dart.dart`.
  - `ExecuTorchModel` with statics `load(String filePath)` and `loadFromBytes(Uint8List modelBytes)`, both returning `Future<ExecuTorchModel>`. No `loadFromAsset`.
  - `ExecutorchManager` with `instance` and every current member **except** `loadModelFromAssets`.
  - `ExecutorchManagerBase` exported publicly.
  - Exported unchanged: `ExecuTorchLLM`, `GenConfig`, `BackendQuery`, `ExecuTorchVersion`, `setNativeDebugLogging`, `Backend`, `ExtendedTensorType`, `ModelLoadResult`, `TensorData`, `TensorType`, `TensorTypeExtension`, `executorchVersion`, the processors, and the error hierarchy.

- [ ] **Step 1: Create the core package skeleton**

```bash
mkdir -p packages/executorch_dart/lib/src
cp LICENSE packages/executorch_dart/LICENSE
cp packages/executorch_flutter/analysis_options.yaml \
   packages/executorch_dart/analysis_options.yaml
```

Then edit `packages/executorch_dart/analysis_options.yaml` and change the first line from `include: package:flutter_lints/flutter.yaml` to:

```yaml
include: package:lints/recommended.yaml
```

`flutter_lints` pulls in Flutter. Leave every other line in that file untouched.

- [ ] **Step 2: Write the core pubspec**

`packages/executorch_dart/pubspec.yaml`:

```yaml
name: executorch_dart
description: >
  ExecuTorch on-device ML inference for pure Dart using dart:ffi — vision
  models plus experimental streaming LLM. Android, iOS, macOS, Linux, Windows.
version: 0.6.0
homepage: https://github.com/abdelaziz-mahdy/executorch_flutter
repository: https://github.com/abdelaziz-mahdy/executorch_flutter

environment:
  sdk: '>=3.6.0 <4.0.0'

resolution: workspace

dependencies:
  code_assets: ^1.0.0
  ffi: ^2.1.4
  hooks: ^1.0.0
  logging: ^1.3.0
  meta: ^1.9.1
  native_toolchain_cmake: ^0.2.2

dev_dependencies:
  ffigen: ^20.1.1
  lints: ^6.0.0
  test: ^1.25.0

topics:
  - machine-learning
  - pytorch
  - ai
  - ffi

hooks:
  user_defines:
    executorch_dart:
      debug: false
      build_mode: "prebuilt"
```

`path_provider`, `flutter_web_plugins`, and `image` are gone on purpose. `path_provider` and `image` had zero references anywhere in the package; `flutter_web_plugins` served only the empty registrant. Do not carry an unused dependency into a brand-new package.

- [ ] **Step 3: Move the core sources**

```bash
cd packages
git mv executorch_flutter/hook executorch_dart/hook
git mv executorch_flutter/native executorch_dart/native
git mv executorch_flutter/test executorch_dart/test
git mv executorch_flutter/ffigen.yaml executorch_dart/ffigen.yaml
git mv executorch_flutter/ffigen_llm.yaml executorch_dart/ffigen_llm.yaml
git mv executorch_flutter/lib/src/build executorch_dart/lib/src/build
git mv executorch_flutter/lib/src/ffi executorch_dart/lib/src/ffi
git mv executorch_flutter/lib/src/generated executorch_dart/lib/src/generated
git mv executorch_flutter/lib/src/processors executorch_dart/lib/src/processors
for f in executorch_errors executorch_inference executorch_llm \
         executorch_manager_base executorch_manager_native \
         executorch_manager_native_stub executorch_manager_unsupported_stub \
         executorch_model executorch_model_ffi_impl executorch_model_ffi_stub \
         executorch_model_unsupported_stub executorch_platform_loader \
         llm_types types version; do
  git mv executorch_flutter/lib/src/$f.dart executorch_dart/lib/src/$f.dart
done
cd ..
```

The web files (`lib/src/web/`, `executorch_llm_web.dart`, `executorch_manager_web_stub.dart`, `executorch_model_web_stub.dart`, `lib/src/ffi/*_web.dart`) stay in `executorch_flutter`. Note `lib/src/ffi/` moved wholesale, taking the three web files with it, and leaving no `lib/src/ffi/` in the wrapper. Recreate that directory before moving them back, or `git mv` fails with "destination directory does not exist":

```bash
cd packages
mkdir -p executorch_flutter/lib/src/ffi
git mv executorch_dart/lib/src/ffi/backend_query_web.dart \
       executorch_flutter/lib/src/ffi/backend_query_web.dart
git mv executorch_dart/lib/src/ffi/version_web.dart \
       executorch_flutter/lib/src/ffi/version_web.dart
git mv executorch_dart/lib/src/ffi/native_logging_web.dart \
       executorch_flutter/lib/src/ffi/native_logging_web.dart
cd ..
```

- [ ] **Step 4: Verify the submodule path again**

```bash
grep -A2 'submodule "native"' .gitmodules
```

Expected: `path = packages/executorch_dart/native`. Fix by hand and run `git submodule sync` if not.

- [ ] **Step 5: Rename the package identifier everywhere in the core**

```bash
cd packages/executorch_dart
grep -rl 'executorch_flutter' lib hook ffigen.yaml ffigen_llm.yaml test \
  | xargs sed -i '' 's|package:executorch_flutter/executorch_flutter.dart|package:executorch_dart/executorch_dart.dart|g; s|package:executorch_flutter/src/|package:executorch_dart/src/|g'
sed -i '' "s|const String _packageName = 'executorch_flutter';|const String _packageName = 'executorch_dart';|" lib/src/build/run_build.dart
cd ../..
```

Then confirm nothing was missed:

```bash
grep -rn 'executorch_flutter' packages/executorch_dart/lib packages/executorch_dart/hook \
  packages/executorch_dart/ffigen.yaml packages/executorch_dart/ffigen_llm.yaml
```

Expected: only log-prefix strings like `'[executorch_flutter] Step 5/5: ...'`. Change those prefixes to `[executorch_dart]` too.

- [ ] **Step 6: Fix the code-asset library name**

`lib/src/build/run_build.dart:428` currently reads:

```dart
    names: {_libraryName: '$_packageName.dart'},
```

That already derives from `_packageName`, so it now resolves to `executorch_dart.dart`. Confirm by reading the line; make no edit if it matches.

- [ ] **Step 7: Write the core public library**

`packages/executorch_dart/lib/executorch_dart.dart`:

```dart
/// ExecuTorch on-device ML inference for pure Dart.
///
/// Runs ExecuTorch models on Android, iOS, macOS, Linux, and Windows through
/// dart:ffi and native assets. Works in any Dart program, including servers
/// and command-line tools — no Flutter required.
///
/// ```dart
/// import 'package:executorch_dart/executorch_dart.dart';
///
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
/// final outputs = await model.forward([inputTensor]);
/// await model.dispose();
/// ```
///
/// Flutter applications should depend on `executorch_flutter` instead, which
/// adds asset-bundle loading and web support on top of this package.
library;

export 'src/executorch_errors.dart';
export 'src/executorch_inference.dart';
export 'src/executorch_llm.dart' show ExecuTorchLLM, GenConfig;
export 'src/executorch_manager_base.dart' show ExecutorchManagerBase;
export 'src/executorch_model.dart';
export 'src/ffi/backend_query.dart' show BackendQuery;
export 'src/ffi/native_logging.dart' show setNativeDebugLogging;
export 'src/ffi/version.dart' show ExecuTorchVersion;
export 'src/processors/processors.dart';
export 'src/types.dart'
    show
        Backend,
        ExtendedTensorType,
        ModelLoadResult,
        TensorData,
        TensorType,
        TensorTypeExtension;
export 'src/version.dart' show executorchVersion;
```

`ExecutorchManagerBase` is exported because the wrapper's web manager extends it, and `implementation_imports` forbids reaching into `src/`.

- [ ] **Step 8: Drop the web branches from the core's conditional imports**

`lib/src/executorch_model.dart` lines 8-11 become:

```dart
import 'executorch_model_unsupported_stub.dart'
    if (dart.library.ffi) 'executorch_model_ffi_stub.dart' as stub;
```

`lib/src/executorch_platform_loader.dart` lines 8-11 become:

```dart
export 'executorch_model_unsupported_stub.dart'
    if (dart.library.ffi) 'executorch_model_ffi_stub.dart';
```

`lib/src/executorch_inference.dart` lines 7-10 become:

```dart
import 'executorch_manager_unsupported_stub.dart'
    if (dart.library.io) 'executorch_manager_native_stub.dart' as stub;
```

- [ ] **Step 9: Remove the asset entry points from the core**

Delete the `loadFromAsset` static from `lib/src/executorch_model.dart` — the doc comment at lines 92-106 and the declaration at 107-108. Also delete the `loadFromAsset` function from `lib/src/executorch_model_ffi_stub.dart` and `lib/src/executorch_model_unsupported_stub.dart`, and from `lib/src/executorch_model_ffi_impl.dart:62`.

In the class doc comment on `ExecuTorchModel`, replace the "Loading from Assets" example block with:

```dart
/// ### Loading from a file (native platforms):
/// ```dart
/// final model = await ExecuTorchModel.load('/path/to/model.pte');
/// final outputs = await model.forward(inputs);
/// await model.dispose();
/// ```
///
/// Flutter applications can load from the asset bundle with
/// `loadModelFromAsset` from `package:executorch_flutter/executorch_flutter.dart`.
```

Also fix the stale reference at `lib/src/executorch_model.dart:61`, which says "On web, use [loadFromAsset] or [loadFromBytes]". `comment_references` is an enabled lint and will error on the dangling link. Replace with "On web, use [loadFromBytes]".

- [ ] **Step 10: Remove `loadModelFromAssets` from the manager**

Delete the declaration at `lib/src/executorch_inference.dart:67` with its doc comment, and the implementation at `lib/src/executorch_manager_base.dart:85-98`. Fix the doc comment at `executorch_inference.dart:57-58`, which references `[loadModelFromAssets]`, to reference `[loadModelFromBytes]` only.

An instance member always wins over an extension member, so leaving this in the core would permanently shadow the wrapper's extension and the Flutter version would never run.

- [ ] **Step 11: Regenerate the FFI bindings**

```bash
cd packages/executorch_dart
dart pub get
dart run ffigen --config ffigen.yaml
dart run ffigen --config ffigen_llm.yaml
grep -n 'DefaultAsset' lib/src/generated/*.g.dart
cd ../..
```

Expected: both files show `@ffi.DefaultAsset('package:executorch_dart/executorch_dart.dart')`.

- [ ] **Step 12: Register the core in the workspace**

In the root `pubspec.yaml`, add to `workspace:`:

```yaml
workspace:
  - packages/executorch_dart
  - packages/executorch_flutter
  - packages/executorch_flutter/example
```

- [ ] **Step 13: Verify the core stands alone**

```bash
dart pub get --directory=packages/executorch_dart
dart analyze packages/executorch_dart 2>&1 | tail -3
(cd packages/executorch_dart && dart test 2>&1 | tail -3)
```

`dart test` has no `--directory` flag — only `dart pub` does. Run it from inside the package.

Expected: `No issues found!` and both tensor conversion tests passing. Then prove no Flutter leaked in:

```bash
grep -rn "package:flutter\|dart:ui\|dart:js_interop" packages/executorch_dart/lib
```

Expected: no output.

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -m "feat: extract pure-Dart executorch_dart core

Moves the FFI layer, native build hook, and native submodule into a
package with no Flutter dependency. Asset loading moves to the wrapper."
```

---

### Task 3: Rebuild the Flutter wrapper

Turn `executorch_flutter` into a thin wrapper: re-export the core, route web, add the asset helpers.

**Files:**
- Modify: `packages/executorch_flutter/pubspec.yaml`
- Rewrite: `packages/executorch_flutter/lib/executorch_flutter.dart`
- Create: `packages/executorch_flutter/lib/src/assets.dart`, `packages/executorch_flutter/test/exports_test.dart`
- Modify: `packages/executorch_flutter/lib/src/web/executorch_model_web.dart`, `lib/src/web/executorch_manager_web.dart`, `lib/src/executorch_llm_web.dart`, `lib/src/ffi/*_web.dart`
- Delete: `packages/executorch_flutter/lib/src/web/executorch_web_plugin.dart`, `lib/src/executorch_model_web_stub.dart`, `lib/src/executorch_manager_web_stub.dart`

**Interfaces:**
- Consumes: everything the core produces in Task 2.
- Produces:
  - `loadModelFromAsset(String assetPath) → Future<ExecuTorchModel>`, top-level.
  - `extension ExecutorchManagerAssets on ExecutorchManager` with `loadModelFromAssets(String assetPath) → Future<ExecuTorchModel>`.
  - On web: classes named `ExecuTorchModel` and `ExecutorchManager` declared by this package, implementing the core interfaces.

- [ ] **Step 1: Rewrite the wrapper pubspec**

```yaml
name: executorch_flutter
description: >
  ExecuTorch on-device ML inference for Flutter using dart:ffi — vision models
  plus experimental streaming LLM. Android, iOS, macOS, Linux, Windows, Web.
version: 0.6.0
homepage: https://github.com/abdelaziz-mahdy/executorch_flutter
repository: https://github.com/abdelaziz-mahdy/executorch_flutter

environment:
  sdk: '>=3.6.0 <4.0.0'
  flutter: '>=3.27.0'

resolution: workspace

dependencies:
  executorch_dart: ^0.6.0
  flutter:
    sdk: flutter
  hooks: ^1.0.0

dev_dependencies:
  flutter_lints: ^6.0.0
  flutter_test:
    sdk: flutter
  integration_test:
    sdk: flutter

flutter:
  plugin:
    platforms:
      android:
        ffiPlugin: true
      ios:
        ffiPlugin: true
      macos:
        ffiPlugin: true
      linux:
        ffiPlugin: true
      windows:
        ffiPlugin: true

executables:
  setup_web: setup_web

topics:
  - machine-learning
  - pytorch
  - ai
  - ffi
  - mobile
```

`hooks` stays because Task 4 adds the tripwire. `flutter_web_plugins` and the `web:` plugin entry are gone: the registrant was an empty method and the wasm files reach an app through `setup_web`, not plugin registration. `path_provider` is gone. `ffi`, `image`, `meta`, `logging`, `code_assets`, `native_toolchain_cmake`, and `ffigen` all belong to the core now.

- [ ] **Step 2: Delete the dead web plugin registrant**

```bash
cd packages/executorch_flutter
git rm lib/src/web/executorch_web_plugin.dart
cd ../..
```

- [ ] **Step 3: Rename the web model class**

In `lib/src/web/executorch_model_web.dart`, change the import of the core model to a prefixed one and rename the class. Line 24 currently reads `class ExecuTorchModelWeb implements ExecuTorchModel {`. It becomes:

```dart
import 'package:executorch_dart/executorch_dart.dart' as core;

/// Web implementation of [core.ExecuTorchModel], backed by WebAssembly.
class ExecuTorchModel implements core.ExecuTorchModel {
```

Keep the existing `loadFromBytes` static. Delete the `loadFromAsset` static at line 95 — asset loading now lives in `lib/src/assets.dart` and calls `loadFromBytes`. Add a `load` static so the web class carries the same statics as the core class:

```dart
  /// Loading from a file path is not supported on web.
  ///
  /// Use `loadModelFromAsset` or [loadFromBytes] instead.
  static Future<ExecuTorchModel> load(String filePath) => throw UnsupportedError(
        'ExecuTorchModel.load() from a file path is not supported on web. '
        'Use loadModelFromAsset() or loadFromBytes() instead.',
      );
```

Rename every other `ExecuTorchModelWeb` reference in the file to `ExecuTorchModel`.

- [ ] **Step 4: Rename the web manager class**

In `lib/src/web/executorch_manager_web.dart`, line 18 currently reads `class ExecutorchManagerWeb extends ExecutorchManagerBase {`. It becomes:

```dart
import 'package:executorch_dart/executorch_dart.dart' as core;

/// Web implementation of the ExecuTorch manager, backed by WebAssembly.
class ExecutorchManager extends core.ExecutorchManagerBase {
```

Rename `_instance` and the `instance` getter to the new type. This class is where `ExecutorchManager.instance` comes from on web, so the static getter must exist here — a wrapper package cannot supply a static onto the core's class.

- [ ] **Step 5: Delete the now-unused web stubs**

```bash
cd packages/executorch_flutter
git rm lib/src/executorch_model_web_stub.dart lib/src/executorch_manager_web_stub.dart
cd ../..
```

The wrapper routes with conditional exports rather than conditional stub imports, so these have no callers.

- [ ] **Step 6: Write the asset helpers**

`packages/executorch_flutter/lib/src/assets.dart`:

```dart
/// Flutter asset-bundle loading for ExecuTorch models.
library;

import 'package:executorch_dart/executorch_dart.dart';
import 'package:flutter/services.dart' show rootBundle;

/// Loads an ExecuTorch model from the Flutter asset bundle.
///
/// Works on every platform, including web. The asset must be declared under
/// `flutter: assets:` in the application's `pubspec.yaml`.
///
/// ```dart
/// final model = await loadModelFromAsset('assets/models/model.pte');
/// ```
///
/// Throws [ExecuTorchException] if the asset is missing or the model fails
/// to load.
Future<ExecuTorchModel> loadModelFromAsset(String assetPath) async {
  final byteData = await rootBundle.load(assetPath);
  return ExecuTorchModel.loadFromBytes(byteData.buffer.asUint8List());
}

/// Asset-bundle loading for [ExecutorchManager].
extension ExecutorchManagerAssets on ExecutorchManager {
  /// Loads a model from the Flutter asset bundle and caches it.
  ///
  /// Equivalent to reading the asset yourself and calling
  /// [ExecutorchManager.loadModelFromBytes].
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    return loadModelFromBytes(byteData.buffer.asUint8List());
  }
}
```

- [ ] **Step 7: Write the wrapper's public library**

Replace the exports at the bottom of `packages/executorch_flutter/lib/executorch_flutter.dart` — keep the existing file-level doc comment, but change the `loadFromAsset` line in its Quick Start example to `loadModelFromAsset`. The export block becomes:

```dart
// Everything from the pure-Dart core except the names routed below.
export 'package:executorch_dart/executorch_dart.dart'
    hide
        BackendQuery,
        ExecuTorchLLM,
        ExecuTorchModel,
        ExecuTorchVersion,
        ExecutorchManager,
        GenConfig,
        setNativeDebugLogging;

// Flutter-only asset helpers.
export 'src/assets.dart';

// Platform routing. On native each line re-exports the core declaration
// unchanged; on web it resolves to this package's implementation. Exactly one
// declaration of each name survives any given compile.
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/executorch_llm_web.dart'
    if (dart.library.js) 'src/executorch_llm_web.dart'
    show ExecuTorchLLM, GenConfig;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/ffi/backend_query_web.dart'
    if (dart.library.js) 'src/ffi/backend_query_web.dart' show BackendQuery;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/ffi/native_logging_web.dart'
    if (dart.library.js) 'src/ffi/native_logging_web.dart'
    show setNativeDebugLogging;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/ffi/version_web.dart'
    if (dart.library.js) 'src/ffi/version_web.dart' show ExecuTorchVersion;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/web/executorch_model_web.dart'
    if (dart.library.js) 'src/web/executorch_model_web.dart'
    show ExecuTorchModel;
export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/web/executorch_manager_web.dart'
    if (dart.library.js) 'src/web/executorch_manager_web.dart'
    show ExecutorchManager;
```

The `hide` list and the routed `show` names must stay identical. A name hidden but not routed vanishes from the Flutter package; a name routed but not hidden collides with the blanket export.

- [ ] **Step 8: Point the web files' core imports at the new package**

Every file left in the wrapper still imports moved core files by relative path. Those paths no longer resolve. Find them all — sibling imports as well as `../` ones:

```bash
cd packages/executorch_flutter
dart analyze lib 2>&1 | grep -E "uri_does_not_exist|Target of URI" | head -20
cd ../..
```

Replace each broken relative import with `import 'package:executorch_dart/executorch_dart.dart';`, adding an `as core` prefix where the file already declares a name that collides.

Known cases, which the analyzer will confirm:

- `lib/src/executorch_llm_web.dart:11` imports `'llm_types.dart'` as a **sibling**. `llm_types.dart` moved to the core, so this needs `package:executorch_dart/executorch_dart.dart`. Check that `GenConfig` is exported from the core library; if `llm_types.dart` holds other types this file uses, add them to the core's `export ... show` list in Task 2 Step 7.
- `lib/src/web/executorch_model_web.dart` imports the core model and `types.dart`.
- `lib/src/web/executorch_manager_web.dart` imports the manager base and `types.dart`.
- `lib/src/ffi/backend_query_web.dart` and `version_web.dart` import `types.dart`.

Re-run the analyze command until it reports no URI errors.

- [ ] **Step 9: Add the export-lockstep test**

Task 2 moved `test/` to the core, so the wrapper has no tests and `flutter test` would fail with "Test directory 'test' not found". More importantly, nothing yet guards the invariant from Step 7: a name hidden but not routed silently disappears from the Flutter package.

Create `packages/executorch_flutter/test/exports_test.dart`:

```dart
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('routed names resolve to exactly one declaration', () {
    // Referencing each name is the assertion: a name hidden from the
    // blanket export but not routed back fails to compile here.
    expect(ExecuTorchModel, isNotNull);
    expect(ExecutorchManager, isNotNull);
    expect(ExecuTorchLLM, isNotNull);
    expect(GenConfig, isNotNull);
    expect(BackendQuery, isNotNull);
    expect(ExecuTorchVersion, isNotNull);
    expect(setNativeDebugLogging, isA<Function>());
  });

  test('Flutter asset helpers are exported', () {
    expect(loadModelFromAsset, isA<Function>());
  });

  test('core types pass through the blanket export', () {
    expect(TensorData, isNotNull);
    expect(TensorType, isNotNull);
    expect(Backend, isNotNull);
    expect(executorchVersion, isNotEmpty);
  });
}
```

These assertions never touch native code, so they need no loaded library and no binding initialization. Their value is compile-time name resolution.

- [ ] **Step 10: Verify the wrapper analyzes and tests**

```bash
flutter pub get
flutter analyze 2>&1 | tail -3
(cd packages/executorch_flutter && flutter test 2>&1 | tail -3)
dart format --set-exit-if-changed packages/executorch_dart/lib \
  packages/executorch_flutter/lib
```

Neither `dart test` nor `flutter test` accepts `--directory` — only `dart pub` does. Run them from inside the package.

Expected: `No issues found!`, tests pass, formatter reports nothing to change. If the formatter rewrites anything, stage it in this task's commit — an unformatted file is how the CI format check fails while local looks clean.

- [ ] **Step 11: Rename the example's `user_defines` key, preserving every value**

Task 2 moved the build hook to `executorch_dart`, so the example's block is now inert. Its settings are not decorative — they select the LLM prebuilt that the Gemma 4 chat screen depends on:

```yaml
hooks:
  user_defines:
    executorch_dart:      # was: executorch_flutter
      debug: true
      llm: true
      backends:
        - xnnpack
        - mlx
```

Change **only** the key. Keep `debug: true`, `llm: true`, both backends, and every explanatory comment in the block exactly as they are. Dropping `llm: true` silently builds a library without the LLM symbols, and the failure appears at runtime in the chat screen, not at build time.

Leave the sibling `dartcv4:` block untouched — it belongs to a different package.

- [ ] **Step 12: Verify the example app still builds against the wrapper**

```bash
cd packages/executorch_flutter/example
flutter build macos --debug 2>&1 | tail -5
cd ../../..
```

Expected: build succeeds. This is the first end-to-end proof that the renamed native asset id resolves through the hook, and that the renamed `user_defines` key is picked up.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat: make executorch_flutter a thin wrapper over executorch_dart

Routes web through conditional exports whose native branch points at the
core, and adds loadModelFromAsset plus the manager asset extension.

BREAKING: ExecuTorchModel.loadFromAsset is now the top-level
loadModelFromAsset."
```

---

### Task 4: `user_defines` tripwire hook

The `user_defines` key must move from `executorch_flutter` to `executorch_dart`, because `input.userDefines` is scoped to the package owning the hook. Left unguarded this fails silently and produces a different backend set with no message. A hook that emits no assets still receives its own package's defines, so the wrapper can catch the stale key and fail the build.

**Files:**
- Create: `packages/executorch_flutter/hook/build.dart`
- Test: manual, via a temporary `user_defines` block in the example pubspec

**Interfaces:**
- Consumes: the wrapper package from Task 3, which already depends on `hooks: ^1.0.0`.
- Produces: a build-time hard failure whenever a consuming app still sets any of the seven legacy keys under `executorch_flutter:`.

- [ ] **Step 1: Write the tripwire hook**

`packages/executorch_flutter/hook/build.dart`:

```dart
// Copyright (c) 2024 ExecuTorch Flutter. All rights reserved.
// Licensed under the MIT license.

import 'package:hooks/hooks.dart';

/// Keys that configured the native build before it moved to executorch_dart.
const _legacyKeys = <String>[
  'build_mode',
  'backends',
  'llm',
  'debug',
  'local_lib_dir',
  'executorch_source',
  'prebuilt_version',
];

void main(List<String> args) async {
  await build(args, (input, output) async {
    final stale = _legacyKeys
        .where((key) => input.userDefines[key] != null)
        .toList(growable: false);
    if (stale.isEmpty) return;
    throw Exception(
      'executorch_flutter no longer owns the native build.\n'
      'Rename this key in your pubspec.yaml:\n'
      '  hooks: user_defines: executorch_flutter:  ->  executorch_dart:\n'
      'Found under the old key: ${stale.join(', ')}',
    );
  });
}
```

This hook produces no assets. It exists only to make a silent misconfiguration loud, and it checks a fixed key list so the wrapper can own real build configuration later without tripping itself.

- [ ] **Step 2: Verify the tripwire fires**

Task 3 renamed the example's key to `executorch_dart:`. To test the tripwire, **temporarily** duplicate that block under the old key — do not edit or delete the real one:

```yaml
hooks:
  user_defines:
    executorch_dart:
      # ... the real block, untouched ...
    executorch_flutter:
      build_mode: "source"
```

Then:

```bash
cd packages/executorch_flutter/example
flutter build macos --debug 2>&1 | grep -i "no longer owns\|Running build hooks failed" | head -3
cd ../../..
```

Expected: the message `executorch_flutter no longer owns the native build.` and a build failure.

- [ ] **Step 3: Remove the temporary block and confirm the build recovers**

Delete the `executorch_flutter:` block you just added, leaving the real `executorch_dart:` block exactly as Task 3 left it — `debug: true`, `llm: true`, `backends: [xnnpack, mlx]`, comments intact.

```bash
cd packages/executorch_flutter/example
git diff pubspec.yaml
flutter build macos --debug 2>&1 | tail -3
cd ../../..
```

Expected: `git diff` reports no changes to the file — you have restored it exactly — and the build succeeds with no tripwire message.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: fail the build when the legacy user_defines key is present

The native build config key moved to executorch_dart. Without this the
rename fails silently and falls back to default backends."
```

---

### Task 5: `dart fix` migration data

Make the one compile-breaking rename automatic.

**Files:**
- Create: `packages/executorch_flutter/lib/fix_data.yaml`
- Test: a throwaway consumer package under the scratch directory

**Interfaces:**
- Consumes: `loadModelFromAsset` from Task 3.
- Produces: `dart fix --apply` rewrites `ExecuTorchModel.loadFromAsset(x)` to `loadModelFromAsset(x)` in consuming code.

- [ ] **Step 1: Write the fix data**

`packages/executorch_flutter/lib/fix_data.yaml`:

```yaml
version: 1
transforms:
  - title: "Use the top-level loadModelFromAsset"
    date: 2026-08-05
    bulkApply: true
    element:
      uris: ['package:executorch_flutter/executorch_flutter.dart']
      method: 'loadFromAsset'
      inClass: 'ExecuTorchModel'
    changes:
      - kind: 'replacedBy'
        newElement:
          uris: ['package:executorch_flutter/executorch_flutter.dart']
          function: 'loadModelFromAsset'
```

- [ ] **Step 2: Build a throwaway consumer that uses the old API**

```bash
mkdir -p /tmp/fixcheck/bin
cat > /tmp/fixcheck/pubspec.yaml <<'EOF'
name: fixcheck
publish_to: none
environment:
  sdk: '>=3.6.0 <4.0.0'
dependencies:
  executorch_flutter:
    path: REPLACE_ME
EOF
```

Replace `REPLACE_ME` with the absolute path to `packages/executorch_flutter`.

```bash
cat > /tmp/fixcheck/bin/main.dart <<'EOF'
import 'package:executorch_flutter/executorch_flutter.dart';

Future<void> main() async {
  final model = await ExecuTorchModel.loadFromAsset('assets/m.pte');
  await model.dispose();
}
EOF
```

- [ ] **Step 3: Verify `dart fix` detects and applies it**

```bash
cd /tmp/fixcheck && dart pub get && dart fix --dry-run 2>&1 | tail -5
```

Expected: a line reporting `undefined_method - 1 fix`.

```bash
cd /tmp/fixcheck && dart fix --apply && cat bin/main.dart
```

Expected: the call reads `await loadModelFromAsset('assets/m.pte');`.

- [ ] **Step 4: Clean up and commit**

```bash
rm -rf /tmp/fixcheck
git add -A
git commit -m "feat: add dart fix migration for the removed loadFromAsset static

Consumers migrate with dart fix --apply instead of by hand."
```

---

### Task 6: Pure-Dart CLI example

This is the proof for issue #42 and nothing existing covers it. It must run with no Flutter anywhere in the dependency graph.

**Files:**
- Create: `packages/executorch_dart/example/pubspec.yaml`, `packages/executorch_dart/example/bin/infer.dart`, `packages/executorch_dart/example/README.md`
- Modify: root `pubspec.yaml`

**Interfaces:**
- Consumes: `ExecuTorchModel.load`, `forward`, `dispose`, `TensorData`, `TensorType`, `ExecuTorchVersion` from the core.
- Produces: a runnable `dart run bin/infer.dart <model.pte>` used by CI in Task 7.

- [ ] **Step 1: Write the example pubspec**

`packages/executorch_dart/example/pubspec.yaml`:

```yaml
name: executorch_dart_example
description: Pure-Dart command-line example for executorch_dart.
publish_to: none

environment:
  sdk: '>=3.6.0 <4.0.0'

resolution: workspace

dependencies:
  executorch_dart: ^0.6.0
```

Note there is no `flutter:` dependency. That absence is the point of the example.

- [ ] **Step 2: Write the CLI**

`packages/executorch_dart/example/bin/infer.dart`:

```dart
// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:executorch_dart/executorch_dart.dart';

Future<void> main(List<String> args) async {
  if (args.length != 1) {
    print('usage: dart run bin/infer.dart <model.pte>');
    exit(64);
  }

  print('ExecuTorch ${ExecuTorchVersion.version}');

  final model = await ExecuTorchModel.load(args.single);
  print('loaded ${model.modelId}');

  final input = TensorData(
    shape: [1, 3, 224, 224],
    dataType: TensorType.float32,
    data: Float32List(1 * 3 * 224 * 224).buffer.asUint8List(),
    name: 'input',
  );

  final outputs = await model.forward([input]);
  print('outputs: ${outputs.length}');
  for (final output in outputs) {
    print('  shape ${output.shape} dtype ${output.dataType}');
  }

  await model.dispose();
  print('ok');
}
```

`ExecuTorchVersion.version` is a static getter on the core class — verified against `lib/src/ffi/version.dart`. Note that the core `ExecuTorchModel` interface exposes only `modelId`, `isDisposed`, `forward`, and `dispose`. It has no `inputShapes`; that member exists only on the web implementation. Do not reach for it here.

- [ ] **Step 3: Write the example README**

`packages/executorch_dart/example/README.md`:

```markdown
# executorch_dart example

Runs an ExecuTorch model from a plain Dart program. No Flutter.

```bash
dart run bin/infer.dart /path/to/mobilenet_v3_xnnpack.pte
```

Compile it to a self-contained bundle, native library included:

```bash
dart build cli
./build/cli/*/bundle/bin/infer /path/to/model.pte
```
```

- [ ] **Step 4: Register it in the workspace**

Add `packages/executorch_dart/example` to the root `pubspec.yaml` `workspace:` list.

- [ ] **Step 5: Run it against a real model**

The `models` submodule at the repo root holds exported `.pte` files. List what is there and pick an XNNPACK-delegated MobileNet matching `executorchVersion` 1.3.1:

```bash
ls models/mobilenet/
```

MobileNet takes `[1, 3, 224, 224]` float32, which is what `bin/infer.dart` already builds. Run it:

```bash
dart pub get
cd packages/executorch_dart/example
dart run bin/infer.dart ../../../models/mobilenet/<the-file-you-listed>.pte
cd ../../..
```

Expected: prints the version, the model id, one or more output shapes, then `ok`.

If the model rejects the input shape, the core interface gives you no way to query the expected one — it has no `inputShapes` member. Read the shape from `models/index.json`, which records it per model, and edit the `shape` list in `bin/infer.dart` to match.

- [ ] **Step 6: Verify the AOT path too**

```bash
cd packages/executorch_dart/example
dart build cli 2>&1 | tail -3
./build/cli/*/bundle/bin/infer ../../../models/mobilenet/<file>.pte
cd ../../..
```

Expected: the bundle runs and prints `ok`. This exercises the same code-asset bundling a server deployment would use.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: add pure-Dart CLI example

Proves inference runs with no Flutter in the dependency graph, under both
dart run and dart build cli."
```

---

### Task 7: CI and packaging metadata

**Files:**
- Modify: `.github/workflows/build.yml`, `deploy-web.yml`, `publish.yml`, `update-readme.yml`, `release.yml`
- Create: `packages/executorch_dart/CHANGELOG.md`, `packages/executorch_dart/.pubignore`
- Move: `packages/executorch_flutter/.pubignore` paths

**Interfaces:**
- Consumes: the CLI example from Task 6, which becomes a CI job.
- Produces: CI that resolves the workspace, analyzes both packages, and publishes the core before the wrapper.

- [ ] **Step 1: Repoint the example working directories**

In `.github/workflows/build.yml` and `.github/workflows/deploy-web.yml`, every `working-directory: example` becomes:

```yaml
        working-directory: packages/executorch_flutter/example
```

- [ ] **Step 2: Repoint the readme-update triggers**

In `.github/workflows/update-readme.yml`, the `paths:` block becomes:

```yaml
    paths:
      - 'packages/executorch_dart/lib/src/build/run_build.dart'
      - 'packages/executorch_dart/lib/src/version.dart'
      - 'packages/executorch_flutter/pubspec.yaml'
```

Its "Extract versions" step reads `pubspec.yaml` at the repo root for `PACKAGE_VERSION`. Change that read to `packages/executorch_flutter/pubspec.yaml`, since the root pubspec is now the unversioned workspace.

- [ ] **Step 3: Publish the core before the wrapper**

In `.github/workflows/publish.yml`, replace the "Install dependencies", "Verify package", and "Publish to pub.dev" steps with:

```yaml
      - name: Install dependencies
        run: flutter pub get

      - name: Verify packages
        run: |
          set +e
          for pkg in packages/executorch_dart packages/executorch_flutter; do
            echo "=== $pkg ==="
            dart pub publish --dry-run --directory="$pkg"
            exit_code=$?
            if [ $exit_code -eq 65 ]; then
              echo "⚠️ $pkg has warnings but can still be published"
            elif [ $exit_code -ne 0 ]; then
              echo "❌ $pkg validation failed (exit code $exit_code)"
              exit $exit_code
            fi
          done

      - name: Publish executorch_dart
        run: dart pub publish --force --directory=packages/executorch_dart

      - name: Publish executorch_flutter
        run: dart pub publish --force --directory=packages/executorch_flutter
```

Order matters: the wrapper declares `executorch_dart: ^0.6.0` and cannot resolve on pub.dev until the core is up.

- [ ] **Step 4: Repoint the release changelog extraction**

In `.github/workflows/release.yml`, the "Extract changelog for version" step reads `CHANGELOG.md` at the repo root. Point it at `packages/executorch_flutter/CHANGELOG.md`.

- [ ] **Step 5: Add a CI job that runs the pure-Dart example**

Append to `.github/workflows/build.yml`:

```yaml
  pure-dart:
    name: Pure Dart (no Flutter)
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Setup Dart
        uses: dart-lang/setup-dart@v1

      - name: Resolve workspace
        run: dart pub get

      - name: Analyze core
        run: dart analyze packages/executorch_dart

      - name: Test core
        run: dart test
        working-directory: packages/executorch_dart

      - name: Build CLI example
        run: dart build cli
        working-directory: packages/executorch_dart/example
```

This job installs Dart only, never Flutter. If it ever needs Flutter to pass, the split has regressed.

- [ ] **Step 6: Give the core a changelog and pubignore**

`packages/executorch_dart/CHANGELOG.md`:

```markdown
## 0.6.0

### Added

- First release. Pure-Dart ExecuTorch inference over dart:ffi, extracted from
  `executorch_flutter` so Dart servers and command-line programs can run models
  without a Flutter SDK. Owns the native build hook and the prebuilt binary
  download.
```

`packages/executorch_dart/.pubignore`:

```
.gitignore
/build/
.dart_tool/
CLAUDE.md
native/CLAUDE.md
tmp/
```

Then trim `packages/executorch_flutter/.pubignore`: drop the `native/`, `models/`, `python/`, and `specs/` entries, which no longer sit inside that package, and repoint the `example/` entries, which still do.

- [ ] **Step 7: Verify both packages pass a publish dry run**

```bash
dart pub publish --dry-run --directory=packages/executorch_dart 2>&1 | tail -5
dart pub publish --dry-run --directory=packages/executorch_flutter 2>&1 | tail -5
```

Expected: each reports either success or exit code 65 with only submodule-related warnings. Any other failure must be fixed before committing.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "ci: build, test, and publish both workspace packages

Adds a Flutter-free job that runs the pure-Dart example on Linux, macOS,
and Windows, and publishes executorch_dart before executorch_flutter."
```

---

### Task 8: Documentation and release notes

**Files:**
- Modify: `packages/executorch_flutter/README.md`, root `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`
- Create: `packages/executorch_dart/README.md`
- Modify: `packages/executorch_flutter/CHANGELOG.md`

**Interfaces:**
- Consumes: every task above.
- Produces: no code. Documentation matching what shipped.

- [ ] **Step 1: Write the core readme**

`packages/executorch_dart/README.md` covers: what the package is, that it needs no Flutter, the `load`/`forward`/`dispose` API, the `hooks: user_defines: executorch_dart:` build configuration block with all seven keys, and a pointer to `executorch_flutter` for Flutter apps and web.

- [ ] **Step 2: Update the wrapper readme**

In `packages/executorch_flutter/README.md`: replace every `ExecuTorchModel.loadFromAsset(` with `loadModelFromAsset(`, change the `user_defines` example key from `executorch_flutter:` to `executorch_dart:`, and add a line pointing Dart-only users at `executorch_dart`.

- [ ] **Step 3: Write the wrapper changelog entry**

Prepend to `packages/executorch_flutter/CHANGELOG.md`:

```markdown
## 0.6.0

### Breaking

- The native build configuration key moved from `executorch_flutter` to
  `executorch_dart`. Rename it under `hooks: user_defines:` in your
  `pubspec.yaml` or the build silently falls back to defaults — the package
  now fails the build with instructions when it finds the old key.
- `ExecuTorchModel.loadFromAsset(...)` is now the top-level
  `loadModelFromAsset(...)`. Run `dart fix --apply` to migrate.
  `manager.loadModelFromAssets(...)` is unchanged.

### Added

- A pure-Dart core, `executorch_dart`, so Dart servers and command-line
  programs can run ExecuTorch models without a Flutter SDK
  ([#42](https://github.com/abdelaziz-mahdy/executorch_flutter/issues/42)).
  `executorch_flutter` is now a thin wrapper over it and keeps asset loading,
  web, and every platform it supported before.
```

Follow the repo's changelog rules: one bullet per user-visible theme, effects rather than diffs, no file paths.

- [ ] **Step 4: Update the root readme**

Extend the package table from Task 1 with an `executorch_dart` row and a one-line note that Flutter apps want `executorch_flutter` while servers want `executorch_dart`.

- [ ] **Step 5: Update `CLAUDE.md`**

The project-structure tree, the "Quick Start" commands, and the ExecuTorch version upgrade procedure all reference the old single-package paths. Update: `flutter analyze lib` becomes workspace-wide, `cd example` becomes `cd packages/executorch_flutter/example`, `lib/src/build/run_build.dart` and `lib/src/version.dart` gain their `packages/executorch_dart/` prefix, and the submodule table notes that `native/` now lives under the core package.

- [ ] **Step 6: Update `CONTRIBUTING.md`**

The build-modes section documents `hooks: user_defines: executorch_flutter:`. Change every occurrence to `executorch_dart:` and update the `native/local-builds/` paths to sit under `packages/executorch_dart/`.

- [ ] **Step 7: Final verification**

```bash
dart pub get
flutter analyze 2>&1 | tail -3
dart format --set-exit-if-changed packages/executorch_dart/lib \
  packages/executorch_flutter/lib
(cd packages/executorch_flutter && flutter test)
(cd packages/executorch_dart && dart test)
git status --short
```

Expected: analyze clean, formatter silent, both test suites pass, and no unstaged formatter changes.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "docs: document the executorch_dart split

Covers the two breaking changes, the new pure-Dart package, and the moved
paths in CLAUDE.md and CONTRIBUTING.md."
```

---

## After the plan

Do not tag a release. CI owns releases: merging to `main` and pushing a `v0.6.0` tag triggers publication, and the core must reach pub.dev before the wrapper — Task 7 Step 3 enforces the order inside the workflow.

Two items the spec flags as unverified, to check before merging:

- **MLX metallib bundling under `dart build cli`.** `run_build.dart:444` hand-bundles `mlx.metallib` because native assets does not carry it. Task 6 Step 6 exercises the AOT bundle; if an MLX-enabled build drops the metallib, fix it there. macOS and LLM-GPU only.
- **Pub workspace behavior with Flutter.** Task 1 Step 8 is the gate. If `flutter pub get` or `flutter analyze` misbehaves at the workspace root, stop and resolve it before Task 2 — every later task assumes workspace resolution works.
