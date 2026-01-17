# ExecuTorch Flutter - Roadmap

This document outlines planned features and improvements for the ExecuTorch Flutter package.

## Completed Features

### Platform Support (All Complete)

- [x] Android support (arm64-v8a, armeabi-v7a, x86_64, x86)
- [x] iOS support (arm64, simulator)
- [x] macOS support (arm64, x86_64)
- [x] Windows desktop support (x64)
- [x] Linux desktop support (x64)
- [x] Web platform support (WebAssembly)

### Architecture Migration

- [x] Migrated from Pigeon to dart:ffi with native assets
- [x] Unified C/C++ FFI layer for all native platforms
- [x] Pre-built binary distribution via GitHub Releases

### Example App

- [x] Live camera integration with real-time inference
- [x] GPU-accelerated preprocessing via Fragment Shaders
- [x] Multiple model support (YOLO, MobileNet)
- [x] Backend selection (XNNPACK, CoreML, MPS)

## Planned Features

### Additional Processors

- [ ] Semantic segmentation processor
- [ ] Instance segmentation processor
- [ ] Text/NLP processors (BERT, GPT tokenizers)
- [ ] Audio classification processors

### Advanced Features

- [ ] Batched inference support
- [ ] Model quantization utilities
- [ ] Async model loading from network
- [ ] Model caching and versioning
- [ ] Performance benchmarking tools

### Documentation

- [ ] Video tutorials
- [ ] Performance optimization guide
- [ ] Troubleshooting guide

## Contributing

Want to work on any of these features? Check out our [Contributing Guide](CONTRIBUTING.md) and feel free to:

1. Open an issue to discuss the feature
2. Submit a pull request with your implementation
3. Share your use cases and requirements

---

**Note**: This roadmap is subject to change based on community feedback and ExecuTorch development.
