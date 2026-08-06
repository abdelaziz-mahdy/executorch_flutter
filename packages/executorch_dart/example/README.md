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
