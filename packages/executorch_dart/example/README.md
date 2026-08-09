# executorch_dart example

Running an ExecuTorch model from a plain Dart program — no Flutter.

## Using the package

Add the dependency:

```yaml
dependencies:
  executorch_dart: ^0.7.0
```

Load a model, run it, dispose it:

```dart
import 'dart:typed_data';

import 'package:executorch_dart/executorch_dart.dart';

Future<void> main() async {
  // Load a .pte file from disk. Use loadFromBytes() if you already have
  // the bytes in memory.
  final model = await ExecuTorchModel.load('mobilenet_v3_small_xnnpack.pte');

  // Build the input tensor. Shape and dtype must match what the model was
  // exported with — MobileNet takes a 224x224 RGB image as float32.
  final input = TensorData(
    shape: [1, 3, 224, 224],
    dataType: TensorType.float32,
    data: Float32List(1 * 3 * 224 * 224).buffer.asUint8List(),
    name: 'input',
  );

  final outputs = await model.forward([input]);

  for (final output in outputs) {
    print('${output.shape} ${output.dataType}');  // [1, 1000] float32
  }

  // Free the native model. Nothing is released automatically.
  await model.dispose();
}
```

That is the whole API: `load` / `loadFromBytes`, `forward`, `dispose`.

Errors arrive as `ExecuTorchException` subclasses, so a real program wraps
the above in a `try` block:

```dart
try {
  // ... load, forward, dispose ...
} on ExecuTorchException catch (e) {
  stderr.writeln('Error: $e');
}
```

Building a Flutter app instead? Use
[`executorch_flutter`](https://pub.dev/packages/executorch_flutter), which
wraps this package and adds asset-bundle loading and Web support.

## Running this example

`bin/infer.dart` is the code above with argument handling. Point it at any
XNNPACK-delegated `.pte`:

```bash
dart run bin/infer.dart /path/to/mobilenet_v3_small_xnnpack.pte
```

```
ExecuTorch 2.0.0
loaded model_1a2b3c
outputs: 1
  shape [1, 1000] dtype TensorType.float32
ok
```

The first run compiles the native library, so expect a delay; later runs are
fast. Compile to a self-contained bundle — native library included — with:

```bash
dart build cli
./build/cli/*/bundle/bin/infer /path/to/model.pte
```

That bundle is what you would deploy to a server.

## Getting a model

`.pte` files are exported from PyTorch. Ready-made ones used by this
project's tests live in the
[models repository](https://github.com/abdelaziz-mahdy/executorch_flutter_models/releases).
Pick an **XNNPACK** build — it is the backend available on every platform.

---

*Copying this example out of the repo? Delete `resolution: workspace` from
its `pubspec.yaml` first; it only resolves inside this repo's pub workspace.
The `executorch_dart` dependency then resolves normally from pub.dev.*
