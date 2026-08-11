## 0.7.1

### Fixed

- macOS Vulkan builds no longer fail with `install_name_tool: ... larger
  updated load commands do not fit`. The bundled `libMoltenVK.dylib` was
  copied from Homebrew, which links it without room to grow its install name,
  so relocating it into an app was impossible. It is now linked from
  MoltenVK's own static library with that headroom reserved, and the build
  refuses to package any bundled library lacking it.
  Thanks @dariyooo ([#51](https://github.com/abdelaziz-mahdy/executorch_flutter/issues/51)).
  - macOS Vulkan variants now require **macOS 12+**, the floor MoltenVK 1.4.x
    is compiled against. Other variants keep their macOS 11 target.

## 0.7.0

### Added

- `Tokenizer` turns text into token ids and back, without a model attached.
  Encoder models — embeddings, classification, retrieval — feed token ids to
  `forward()` and have no generation loop to borrow one from, so until now
  there was no way to tokenize for them. Supports HuggingFace `tokenizer.json`
  built on BPE, SentencePiece, TikToken and llama2.c, detected automatically.
  Thanks @dariyooo ([#45](https://github.com/abdelaziz-mahdy/executorch_flutter/issues/45)).
  - WordPiece/BERT-family tokenizers are **not** supported: the ExecuTorch
    reader is BPE-only and implements no `BertNormalizer`, which rules out
    BERT, DistilBERT, MiniLM and most sentence-transformers models. Passing one
    raises an error naming the specific reason rather than a generic parse
    failure.
  - Not available on Web, which has no `dart:ffi`.

### Changed

- Upgraded to ExecuTorch 1.4.0.

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
