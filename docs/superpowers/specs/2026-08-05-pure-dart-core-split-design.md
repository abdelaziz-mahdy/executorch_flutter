# Pure-Dart core split: `executorch_dart` + `executorch_flutter`

**Date:** 2026-08-05
**Issue:** [#42 — Splitting out a pure-Dart core](https://github.com/abdelaziz-mahdy/executorch_flutter/issues/42)
**Status:** design approved, implementation plan pending

## Goal

Let non-Flutter Dart programs — servers above all — run ExecuTorch inference. Today
`executorch_flutter` forces a Flutter SDK dependency on every consumer, though it uses
Flutter for almost nothing.

Split the package in two:

- `executorch_dart` — pure Dart, native platforms, owns the FFI layer and the native build.
- `executorch_flutter` — thin Flutter wrapper: asset loading, web, and the plugin manifest.

## Verified facts

Each of these was tested on Dart 3.12.2 stable, not assumed.

**Build hooks work outside Flutter.** A pure-Dart package with `hook/build.dart`,
`@DefaultAsset`, and `@Native` resolves its symbols under `dart run` (JIT), `dart build cli`
(AOT — the dylib lands in the bundle and the binary runs), and `dart test`. No experiment
flag. This is the premise the whole issue rests on, and it holds.

**Extensions add instance members, never statics.** `extension X on C { void m() {} }` lets
another package add `c.m()`. No extension form makes `C.m()` resolve — not today, and not
under `--enable-experiment=static-extensions`, which the SDK accepts as a flag but does not
implement. Both cases fail with `undefined_method`.

**A hook that emits no assets still reads its own `user_defines` and can fail the build.**
This makes the tripwire below possible.

**`dart fix` migrates a removed static.** A `fix_data.yaml` `replacedBy` transform rewrote
`A.loadFromAsset('x')` into `loadModelFromAsset('x')` and the result analyzed clean.

## Repository layout

```
executorch_flutter/                  # repo root = pub workspace only
├── pubspec.yaml                     # publish_to: none, workspace: [...]
├── models/                          # submodule, stays at root
└── packages/
    ├── executorch_dart/             # published — pure Dart, native only
    │   ├── lib/  hook/  native/     # native submodule moves here
    │   ├── ffigen.yaml  ffigen_llm.yaml  test/
    │   └── example/                 # new: pure-Dart CLI
    └── executorch_flutter/          # published — Flutter wrapper
        ├── lib/  hook/              # hook is a tripwire only
        ├── web/  bin/setup_web.dart
        ├── android/ ios/ macos/ linux/ windows/
        └── example/                 # today's example/
```

The `native/` submodule must sit inside the core package: the build hook resolves
`getPackagePath(_packageName)` and then `packagePath.uri.resolve('native/')`
(`run_build.dart:101,333`). Move it with `git mv` and update `.gitmodules`.

Example path dependencies need no edits. `example/` moves to
`packages/executorch_flutter/example/`, so its `path: ../` still points at the wrapper.

## Package boundaries

**`executorch_dart` takes** the FFI layer, generated bindings, build hook, LLM, manager,
processors, types, and errors. It imports `dart:ffi` and `dart:io` and nothing else.

**`executorch_flutter` keeps** three things:

1. `loadModelFromAsset` and the `ExecutorchManager` asset extension, both using `rootBundle`
2. the entire web implementation — js_interop, wasm loader, `web/`, `bin/setup_web.dart`
3. the plugin platform directories and the pubspec `flutter: plugin:` block

**Deleted:** the `path_provider` dependency, which nothing references, and
`flutter_web_plugins` along with `lib/src/web/executorch_web_plugin.dart`, whose
`registerWith` is an empty method. The wasm files reach an app through `setup_web`, not
through plugin registration.

### Why web stays in the wrapper

> **Superseded premise.** The paragraph below leads with a `rootBundle` dependency that no
> longer exists in the web implementation. The decision it argues for still stands — see
> "Amendment: updated rationale" at the end of this section for the reasons that now carry
> it.

The web implementation already depends on Flutter — `executorch_model_web.dart:97` calls
`rootBundle`. Leaving it wrapper-side avoids inventing an asset abstraction for it, keeps
6 MB of wasm out of a server dependency, and preserves `dart run executorch_flutter:setup_web`
unchanged.

The wrapper routes platforms with conditional export lines whose native branch re-exports the
core class directly:

```dart
export 'package:executorch_dart/executorch_dart.dart'
    hide ExecuTorchModel, ExecutorchManager, ExecuTorchLLM,
         BackendQuery, ExecuTorchVersion, setNativeDebugLogging;

export 'package:executorch_dart/executorch_dart.dart'
    if (dart.library.js_interop) 'src/web/llm_web.dart' show ExecuTorchLLM;
// ... one such line per routed name
```

This is routing, not shadowing, and the distinction carries the whole design. Exactly one
declaration of each name survives any given compile: on native every routed line resolves to
the core class, so core-typed APIs and user-typed variables agree. On web the routed line
resolves to a wrapper class that `implements` the core interface, which keeps it assignable
wherever core expects the interface.

The blanket `hide` list and the routed lines must stay in lockstep. A name hidden but not
routed disappears from the Flutter package; a name routed but not hidden collides.

The cost is that a pure-Dart web program — dart2js without Flutter — cannot use this package.
No such consumer exists.

#### Amendment: updated rationale (2026-08-06)

The `rootBundle` premise the section above leads with no longer holds. `rootBundle` moved
out of the web implementation and into `lib/src/assets.dart` (`loadModelFromAsset` and
`ExecutorchManagerAssets`). Every file under `lib/src/web/`, plus the web-routed
`lib/src/ffi/*_web.dart` files and `lib/src/executorch_llm_web.dart`, import no
`package:flutter/*` at all — the web implementation itself is pure Dart and `dart:js_interop`
today.

The decision to keep web in the wrapper stands anyway, on different grounds:

- **Payload and package identity.** `web/wasm/` holds ~5 MB of wasm (`executorch.wasm`
  3 MB, `executor_runner.wasm` 2 MB, per `dart pub publish --dry-run`) — nearly the entire
  wrapper archive. The core publishes at 109 KB. Moving web into the core would put that
  wasm into the package whose entire purpose is being a lightweight server dependency, and
  would make `executorch_dart` a web package on pub.dev — contradicting its own README.
- **Entry-point stability.** Moving web would also rename the public
  `dart run executorch_flutter:setup_web` entry point.
- **The routing block's cost is bounded.** The invariant it depends on — the blanket `hide`
  list and the routed `show` names must stay identical — is documented in three places (this
  section, the inline comments in `packages/executorch_flutter/lib/executorch_flutter.dart`,
  and `packages/executorch_dart/test/library_split_test.dart`) and is now guarded by tests on
  both sides: `packages/executorch_dart/test/library_split_test.dart` and
  `packages/executorch_flutter/test/library_split_test.dart`.

**Revisit trigger.** This gets more expensive after publish: `executorch_dart_shared.dart`
becomes a public library and the routing shape becomes an API commitment, so changing it
later is a breaking change for both packages rather than a pre-release refactor.

## API design

On native platforms `ExecuTorchModel` is the core class, unwrapped. The wrapper adds only the
asset entry points:

```dart
// packages/executorch_flutter/lib/executorch_flutter.dart
Future<ExecuTorchModel> loadModelFromAsset(String assetPath) async =>
    ExecuTorchModel.loadFromBytes(
        (await rootBundle.load(assetPath)).buffer.asUint8List());

extension ExecutorchManagerAssets on ExecutorchManager {
  Future<ExecuTorchModel> loadModelFromAssets(String assetPath) async =>
      loadModelFromBytes((await rootBundle.load(assetPath)).buffer.asUint8List());
}
```

`manager.loadModelFromAssets(...)` survives verbatim, because extension instance members
work. The static becomes a top-level function:

```dart
- await ExecuTorchModel.loadFromAsset('assets/models/m.pte');
+ await loadModelFromAsset('assets/models/m.pte');
```

This design is compile-safe in both directions. A server programmer who types
`ExecuTorchModel.loadFromAsset` gets an analyzer error, never a runtime surprise.

### Statics force the web classes to be real declarations

`ExecuTorchModel.load`, `ExecuTorchModel.loadFromBytes`, and `ExecutorchManager.instance` are
all static. A wrapper package cannot override or supply a static on a class it imports, so the
web branch of each routed name must be a class the wrapper *declares*, named exactly like the
core one and carrying its own statics:

```dart
// packages/executorch_flutter/lib/src/web/model_web.dart
class ExecuTorchModel implements core.ExecuTorchModel {
  static Future<ExecuTorchModel> loadFromBytes(Uint8List bytes) => ...;
}
```

Miss this and `ExecutorchManager.instance` throws on Flutter web, which the example app calls
at startup (`example/lib/main.dart:36`) and CI deploys through `deploy-web.yml`.

### Why not a façade

A wrapper class named `ExecuTorchModel` that re-declared the three statics would preserve the
old call syntax, but `ExecutorchManager.loadModel()` returns the *core* type. Hiding the core
type behind a façade of the same name breaks every manager call site:

```dart
final ExecuTorchModel m = await manager.loadModel(path);  // core type, façade variable
```

Fixing that means façading `ExecutorchManager` too — thirteen members — and re-wrapping every
model it returns, forever. Every core API added later would need a matching façade edit or it
would silently go missing from the Flutter package.

The ecosystem agrees. `drift_flutter` wraps the pure-Dart `drift` with a single top-level
function and re-exports core types; `sqflite_common_ffi` wraps `sqlite3`; `media_kit_video`
wraps `media_kit`. None of them shadows a core type. Flutter's federated-plugin guidance
covers method-channel splits and does not apply to an FFI package on native assets, where the
`package_ffi` build-hook template is the recommended shape.

### Core must drop `loadModelFromAssets`

Remove it from the `ExecutorchManager` interface and from `ExecutorchManagerBase`
(`executorch_inference.dart:67`, `executorch_manager_base.dart:85`). An instance member always
beats an extension member, so leaving it in core would shadow the wrapper's extension and the
Flutter version would never run.

## Breaking changes

Two, both guarded by mechanism rather than by documentation.

### 1. `user_defines` key rename — guarded by a tripwire

Apps must rename the key, because `input.userDefines` is scoped to the package that owns the
hook (`run_build.dart:106`):

```yaml
hooks:
  user_defines:
    executorch_dart:      # was: executorch_flutter
      build_mode: "source"
```

Left alone this fails silently — the build falls back to defaults and produces a different
backend set with no message. So `executorch_flutter` keeps a `hook/build.dart` that emits no
assets and exists only to catch the legacy block:

```
Error: executorch_flutter no longer owns the native build.
Rename this key in your pubspec.yaml:
  hooks: user_defines: executorch_flutter:  ->  executorch_dart:
(found: build_mode, backends)
```

The tripwire fires only on the known legacy keys — `build_mode`, `backends`, `llm`, `debug`,
`local_lib_dir`, `executorch_source`, `prebuilt_version` — so the wrapper can own real build
configuration later without fighting itself. It stays in place permanently; one cached no-op
hook run costs little, and the rename has no expiry date.

### 2. `ExecuTorchModel.loadFromAsset` — guarded by `dart fix`

`packages/executorch_flutter/lib/fix_data.yaml` ships a `replacedBy` transform, so migration is
`dart fix --apply`. A consumer who skips it gets a compile error.

### Everything else holds

Imports of `package:executorch_flutter/executorch_flutter.dart` keep working.
`manager.loadModelFromAssets(...)` keeps working. `dart run executorch_flutter:setup_web` keeps
working. Only deep imports of `package:executorch_flutter/src/...` break, and those were never
public.

## Build hook changes

All inside the core package:

- `_packageName` becomes `executorch_dart` (`run_build.dart:71`)
- the code-asset name becomes `executorch_dart.dart` (`run_build.dart:428`)
- `@DefaultAsset` becomes `package:executorch_dart/executorch_dart.dart` in both generated files
- update `ffigen.yaml`, `ffigen_llm.yaml`, and `executorch_symbols.yaml`, then regenerate

## CI and release

| workflow | change |
|---|---|
| `build.yml`, `deploy-web.yml` | `working-directory: example` becomes `packages/executorch_flutter/example` |
| `update-readme.yml` | trigger paths become `packages/executorch_dart/lib/src/{build/run_build,version}.dart` |
| `publish.yml` | two sequential publishes via `--directory`, core first |
| `release.yml` | changelog extraction points at the wrapper's `CHANGELOG.md` |

Each package carries its own `CHANGELOG.md` and `.pubignore`, as pub.dev requires. The
pre-push checklist — `flutter analyze lib` and `dart format --set-exit-if-changed lib` — becomes
workspace-wide.

Both packages ship 0.6.0 together. The wrapper declares `executorch_dart: ^0.6.0`, which pub
resolves to the local workspace member during development and to pub.dev after publish.

## Testing

- Move the two existing unit tests to core.
- Keep the five integration tests in `example/integration_test/` with the Flutter example.
- Add a pure-Dart CLI example under `packages/executorch_dart/example` that loads a `.pte` and
  runs `forward` with no Flutter in the dependency graph. Run it in CI on Linux, macOS, and
  Windows. This is the proof for #42 and nothing existing covers it.

## Risks and open items

**The `models` submodule is deleted in the working tree.** Restore it before any submodule
surgery, or the `native/` move lands in a dirty tree.

**MLX metallib bundling is unverified for `dart build cli`.** `run_build.dart:444` hand-bundles
`mlx.metallib` for Flutter because native assets does not carry it. The AOT-bundle path needs
checking. macOS and LLM-GPU only.

**Pub workspace plus Flutter is new ground here.** Confirm `flutter pub get`, `flutter analyze`,
and `flutter test` behave at the workspace root before rewiring CI.
