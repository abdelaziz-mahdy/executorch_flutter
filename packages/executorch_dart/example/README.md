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

**Copying this example out of the repo?** `pubspec.yaml` pins
`resolution: workspace`, which only resolves inside this repo's pub
workspace. Delete that line first — the existing `executorch_dart: ^0.6.0`
dependency then resolves normally from pub.dev.
