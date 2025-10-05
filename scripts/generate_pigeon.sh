#!/bin/bash
# Pigeon code generation script for iOS and macOS platforms
#
# This script generates Pigeon code for iOS and copies it to macOS.
# The macOS version needs modifications because:
# 1. SPM requires public visibility for types used in public APIs
# 2. FlutterError needs Error protocol conformance on macOS
#
# Usage: ./scripts/generate_pigeon.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Pigeon code generation...${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."

echo -e "${BLUE}[1/3] Generating Pigeon code for iOS/Android/Dart...${NC}"
flutter pub run pigeon --input pigeons/executorch_api.dart

echo -e "${BLUE}[2/3] Applying SPM patches to iOS generated code...${NC}"
# iOS also uses SPM, so it needs the same patches as macOS
# Add FlutterError conformance after imports
sed -i '' '/#endif/a\
\
/// Make FlutterError conform to Error protocol (required for SPM)\
extension FlutterError: Error {}\
' ios/Classes/Generated/ExecutorchApi.swift

# Make types public (SPM requirement - types used in public APIs must be public)
sed -i '' 's/^enum /public enum /g' ios/Classes/Generated/ExecutorchApi.swift
sed -i '' 's/^struct /public struct /g' ios/Classes/Generated/ExecutorchApi.swift
sed -i '' 's/^protocol /public protocol /g' ios/Classes/Generated/ExecutorchApi.swift

echo -e "${BLUE}[3/3] Copying patched Swift code to SPM locations...${NC}"
# Both iOS and macOS use SPM, so they share the same generated file
mkdir -p macos/executorch_flutter/Sources/executorch_flutter/Generated
mkdir -p ios/executorch_flutter/Sources/executorch_flutter/Generated

cp ios/Classes/Generated/ExecutorchApi.swift \
   macos/executorch_flutter/Sources/executorch_flutter/Generated/ExecutorchApi.swift

cp ios/Classes/Generated/ExecutorchApi.swift \
   ios/executorch_flutter/Sources/executorch_flutter/Generated/ExecutorchApi.swift

echo -e "${GREEN}✓ Pigeon generation complete!${NC}"
echo -e "${GREEN}  • Dart: lib/src/generated/executorch_api.dart${NC}"
echo -e "${GREEN}  • Kotlin: android/src/main/kotlin/.../ExecutorchApi.kt${NC}"
echo -e "${GREEN}  • iOS: ios/Classes/Generated/ExecutorchApi.swift${NC}"
echo -e "${GREEN}  • macOS: macos/.../Generated/ExecutorchApi.swift (SPM-compatible)${NC}"
