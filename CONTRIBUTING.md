# Contributing to ExecuTorch Flutter

Thank you for your interest in contributing to ExecuTorch Flutter! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Submitting Changes](#submitting-changes)
- [Platform-Specific Guidelines](#platform-specific-guidelines)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/executorch_flutter.git
   cd executorch_flutter
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/abdelaziz-mahdy/executorch_flutter.git
   ```

## Development Setup

### Prerequisites

- Flutter SDK 3.38+ (first version with native assets hooks)
- Dart SDK 3.0.0 or later
- **Android Development**:
  - Android Studio with SDK API 23+
  - NDK for native development
- **iOS Development**:
  - macOS with Xcode 14+
  - iOS 13.0+ (device and simulator supported)
- **macOS Development**:
  - macOS 11.0+ (Big Sur or later)
  - Apple Silicon or Intel Mac
- **Windows Development**:
  - Windows 10+
  - Visual Studio 2022 with C++ workload
- **Linux Development**:
  - Ubuntu 20.04+ or equivalent
  - GCC/Clang, CMake, ninja-build

### Setup Steps

1. **Install dependencies**:
   ```bash
   flutter pub get
   ```

2. **Run the example app**:
   ```bash
   cd example
   flutter run
   ```

## Project Structure

```
executorch_flutter/
├── lib/                        # Dart library code
│   ├── src/
│   │   ├── ffi/                # dart:ffi bindings and FFI layer
│   │   ├── generated/          # ffigen-generated FFI bindings
│   │   ├── build/              # Native assets build hook
│   │   ├── executorch_model.dart
│   │   ├── executorch_inference.dart
│   │   └── executorch_errors.dart
│   └── executorch_flutter.dart # Public API exports
├── native/                     # Git submodule: C/C++ FFI library
│   ├── src/                    # FFI implementation
│   ├── cmake/                  # CMake build configuration
│   └── scripts/                # Platform build scripts
└── example/                    # Example Flutter app
```

## Making Changes

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for major changes to discuss the approach
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Workflow

1. Make your changes following the [Code Standards](#code-standards)
2. Test your changes thoroughly on the example app
3. Update documentation as needed
4. Commit your changes with clear messages (see below)

### Commit Message Format

Follow the conventional commits format:

```
type(scope): brief description

Detailed description if needed

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no functional changes)
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Examples:**
```
feat(android): add support for XNNPACK backend
fix(ios): resolve memory leak in model disposal
docs(readme): update installation instructions
```

## Code Standards

### Dart Code

- Follow the [Dart Style Guide](https://dart.dev/guides/language/effective-dart/style)
- Run `dart format .` before committing
- Run `dart analyze` and fix all warnings
- Use meaningful variable and function names
- Add documentation comments for public APIs

### Kotlin Code (Android)

- Follow [Kotlin coding conventions](https://kotlinlang.org/docs/coding-conventions.html)
- Use Android Studio's built-in formatter
- Handle errors gracefully with try-catch blocks
- Use coroutines for async operations

### Swift Code (iOS/macOS)

- Follow [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)
- Use SwiftFormat for consistent formatting
- Use `async/await` for asynchronous operations
- Utilize Swift actors for thread safety

### FFI API Changes

If modifying the FFI layer:

1. **Update** C/C++ code in `native/src/`
2. **Update** header files in `native/src/executorch_ffi.h`
3. **Regenerate** FFI bindings:
   ```bash
   dart run ffigen
   ```
4. **Test** changes using integration tests (see below)

### Integration Testing

After making changes, run the integration tests:

```bash
cd example
flutter test integration_test/models_integration_test.dart -d macos   # macOS
flutter test integration_test/models_integration_test.dart -d ios     # iOS
flutter test integration_test/models_integration_test.dart -d android # Android
flutter test integration_test/models_integration_test.dart -d windows # Windows
flutter test integration_test/models_integration_test.dart -d linux   # Linux
```

**Prerequisites**:
- Models are automatically downloaded from GitHub on first use
- To export models manually: `cd models/python && python3 main.py`

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues
   - Screenshots/videos for UI changes
   - Verification that changes work on affected platforms

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings from analyzer
- [ ] Verified on example app
- [ ] Works on all affected platforms

### Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Keep the PR updated with main branch
- Once approved, maintainers will merge

## Platform-Specific Guidelines

### Android

- **Minimum SDK**: API 23
- **Target Architecture**: arm64-v8a
- **Dependencies**: Use Gradle for dependency management
- **ExecuTorch**: Version 1.0.0-rc2 via AAR (`org.pytorch:executorch-android:1.0.0-rc2`)

### iOS

- **Minimum Version**: iOS 13.0
- **Architectures**: arm64 (device), arm64-simulator, x86_64-simulator
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK, CoreML (optional)

### macOS

- **Minimum Version**: macOS 11.0 (Big Sur)
- **Architectures**: arm64 (Apple Silicon), x86_64 (Intel)
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK, CoreML, MPS (Metal Performance Shaders)

### Windows

- **Minimum Version**: Windows 10
- **Architecture**: x64
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK

### Linux

- **Minimum Version**: Ubuntu 20.04+ or equivalent
- **Architecture**: x64
- **Build System**: Native assets via CMake
- **Backend**: XNNPACK

## Need Help?

- 🐛 [GitHub Issues](https://github.com/abdelaziz-mahdy/executorch_flutter/issues) - Report bugs or ask questions
- 📖 [Documentation](https://github.com/abdelaziz-mahdy/executorch_flutter) - Read the docs

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

---

Thank you for contributing to ExecuTorch Flutter! 🎉
