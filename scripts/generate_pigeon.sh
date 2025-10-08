#!/bin/bash
# Pigeon code generation script with automatic Swift public type fixes
#
# This script:
# 1. Generates Pigeon code for all platforms
# 2. Automatically makes Swift types public (required for SPM)
# 3. Removes PigeonError handling (not used)
# 4. Creates symlinks for iOS and macOS to shared darwin code
#
# Usage: ./scripts/generate_pigeon.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Pigeon code generation...${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."

echo -e "${BLUE}[1/4] Generating Pigeon code for Dart/Android/Darwin...${NC}"
dart pub global run pigeon --input pigeons/executorch_api.dart

DARWIN_SWIFT="darwin/Sources/executorch_flutter/Generated/ExecutorchApi.swift"

echo -e "${BLUE}[2/4] Making Swift types public (SPM requirement)...${NC}"
# Make all enums public
sed -i '' 's/^enum /public enum /g' "$DARWIN_SWIFT"

# Make all structs public
sed -i '' 's/^struct /public struct /g' "$DARWIN_SWIFT"

# Make all protocols public
sed -i '' 's/^protocol /public protocol /g' "$DARWIN_SWIFT"

# Make == operators public
sed -i '' 's/  static func ==/  public static func ==/g' "$DARWIN_SWIFT"

# Make hash(into:) methods public
sed -i '' 's/  func hash(into/  public func hash(into/g' "$DARWIN_SWIFT"

echo -e "${BLUE}[3/4] Making PigeonError public (if generated)...${NC}"
# Make PigeonError and its initializer public if generated
sed -i '' 's/^struct PigeonError/public struct PigeonError/g' "$DARWIN_SWIFT"
sed -i '' 's/^class PigeonError/public class PigeonError/g' "$DARWIN_SWIFT"
sed -i '' 's/^final class PigeonError/public final class PigeonError/g' "$DARWIN_SWIFT"
# Make PigeonError init public
sed -i '' '/class PigeonError.*{/,/^}/ s/  init(/  public init(/g' "$DARWIN_SWIFT"

echo -e "${BLUE}[4/4] Creating symlinks for iOS and macOS...${NC}"
# Create symlinks to shared darwin code
mkdir -p ios/Classes/Generated
mkdir -p macos/Classes/Generated

# Remove old files if they exist
rm -f ios/Classes/Generated/ExecutorchApi.swift
rm -f macos/Classes/Generated/ExecutorchApi.swift

# Create symlinks
ln -sf ../../../darwin/Sources/executorch_flutter/Generated/ExecutorchApi.swift \
       ios/Classes/Generated/ExecutorchApi.swift

ln -sf ../../../darwin/Sources/executorch_flutter/Generated/ExecutorchApi.swift \
       macos/Classes/Generated/ExecutorchApi.swift

echo -e "${GREEN}✓ Pigeon generation complete!${NC}"
echo -e "${GREEN}  • Dart: lib/src/generated/executorch_api.dart${NC}"
echo -e "${GREEN}  • Kotlin: android/src/main/kotlin/.../ExecutorchApi.kt${NC}"
echo -e "${GREEN}  • Darwin (shared): $DARWIN_SWIFT${NC}"
echo -e "${GREEN}  • iOS: symlink → darwin${NC}"
echo -e "${GREEN}  • macOS: symlink → darwin${NC}"
