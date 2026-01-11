#!/bin/bash

# Integration Test Runner for ExecuTorch Flutter Example App
# This script runs integration tests on all supported platforms
# Falls back to building if devices/simulators are not accessible
#
# Usage:
#   ./run_integration_tests.sh           # Run tests on all platforms
#   ./run_integration_tests.sh macos     # Run tests only on macOS
#   ./run_integration_tests.sh ios       # Run tests only on iOS
#   ./run_integration_tests.sh android   # Run tests only on Android

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse platform argument
TARGET_PLATFORM="${1:-all}"

# Validate platform argument
if [[ ! "$TARGET_PLATFORM" =~ ^(all|macos|ios|android)$ ]]; then
    echo "Error: Invalid platform '$TARGET_PLATFORM'"
    echo "Usage: $0 [all|macos|ios|android]"
    echo ""
    echo "Examples:"
    echo "  $0          # Run tests on all platforms (default)"
    echo "  $0 macos    # Run tests only on macOS"
    echo "  $0 ios      # Run tests only on iOS"
    echo "  $0 android  # Run tests only on Android"
    exit 1
fi

echo "🧪 ExecuTorch Flutter Integration Tests"
echo "========================================"
if [ "$TARGET_PLATFORM" != "all" ]; then
    echo "Target Platform: $TARGET_PLATFORM"
fi
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track test results
ANDROID_RESULT=""
IOS_RESULT=""
MACOS_RESULT=""

# Function to run tests on a platform
run_tests() {
    local platform=$1
    local device_flag=$2

    echo ""
    echo "${YELLOW}Running integration tests on $platform...${NC}"
    echo "----------------------------------------"

    if flutter test integration_test/models_integration_test.dart $device_flag; then
        echo "${GREEN}✓ $platform tests PASSED${NC}"
        return 0
    else
        echo "${RED}✗ $platform tests FAILED${NC}"
        return 1
    fi
}

# Function to build for a platform (fallback when no device)
build_platform() {
    local platform=$1
    local build_command=$2

    echo ""
    echo "${BLUE}📦 Building for $platform (no device available)...${NC}"
    echo "----------------------------------------"

    if eval "$build_command"; then
        echo "${GREEN}✓ $platform build SUCCESSFUL${NC}"
        return 0
    else
        echo "${RED}✗ $platform build FAILED${NC}"
        return 1
    fi
}

# Note: Models are downloaded from GitHub at runtime
echo "📦 Model Download Info"
echo "   Models will be downloaded from GitHub during tests."
echo "   Repository: https://github.com/abdelaziz-mahdy/executorch_flutter_models"
echo ""
echo "   Required models:"
echo "   - mobilenet_v3_small_xnnpack.pte"
echo "   - yolo11n_xnnpack.pte"
echo "   - yolov5n_xnnpack.pte"
echo "   - yolov8n_xnnpack.pte"
echo ""

# Helper function to get flutter devices (cached to avoid broken pipe issues)
get_flutter_devices() {
    flutter devices 2>/dev/null || true
}

# Test on macOS (if on macOS)
if [[ "$TARGET_PLATFORM" == "all" || "$TARGET_PLATFORM" == "macos" ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "📍 Detected macOS platform"

        # Check if macOS device is available
        DEVICES_OUTPUT=$(get_flutter_devices)
        if echo "$DEVICES_OUTPUT" | grep -q "macos"; then
            if run_tests "macOS" "-d macos"; then
                MACOS_RESULT="✓ PASSED (tests)"
            else
                MACOS_RESULT="✗ FAILED (tests)"
            fi
        else
            # No macOS device, fallback to build
            echo "${YELLOW}⚠️  macOS device not available, building instead...${NC}"
            if build_platform "macOS" "flutter build macos --debug"; then
                MACOS_RESULT="✓ BUILD OK"
            else
                MACOS_RESULT="✗ BUILD FAILED"
            fi
        fi
    else
        echo "⚠️  Skipping macOS (not on macOS)"
        MACOS_RESULT="⊘ SKIPPED (not macOS)"
    fi
else
    MACOS_RESULT="⊘ SKIPPED (not selected)"
fi

# Test on iOS (physical device only - simulator not supported)
if [[ "$TARGET_PLATFORM" == "all" || "$TARGET_PLATFORM" == "ios" ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo "📍 Checking for iOS devices..."

        IOS_DEVICE=""
        # Check if any iOS physical device is connected
        DEVICES_OUTPUT=$(get_flutter_devices)
        if echo "$DEVICES_OUTPUT" | grep -q "ios" && ! echo "$DEVICES_OUTPUT" | grep "ios" | grep -q "simulator"; then
            # Extract device ID from flutter devices output (format: "name • device_id • platform • details")
            IOS_DEVICE=$(echo "$DEVICES_OUTPUT" | grep "ios" | grep -v "simulator" | head -1 | sed -E 's/.*• ([^ ]+) *• ios.*/\1/')
            echo "Found iOS physical device: $IOS_DEVICE"

            if run_tests "iOS" "-d $IOS_DEVICE"; then
                IOS_RESULT="✓ PASSED (tests)"
            else
                IOS_RESULT="✗ FAILED (tests)"
            fi
        else
            # No physical device, fallback to build
            echo "${YELLOW}⚠️  No iOS physical device connected${NC}"
            echo "${YELLOW}⚠️  iOS Simulator is NOT supported (ExecuTorch requires arm64)${NC}"
            echo "${BLUE}📦 Building iOS app for device instead...${NC}"

            if build_platform "iOS" "flutter build ios --release --no-codesign"; then
                IOS_RESULT="✓ BUILD OK (arm64)"
            else
                IOS_RESULT="✗ BUILD FAILED"
            fi
        fi
    else
        echo "⚠️  Skipping iOS (not on macOS)"
        IOS_RESULT="⊘ SKIPPED (not macOS)"
    fi
else
    IOS_RESULT="⊘ SKIPPED (not selected)"
fi

# Test on Android (use emulator or device)
if [[ "$TARGET_PLATFORM" == "all" || "$TARGET_PLATFORM" == "android" ]]; then
    echo ""
    echo "📍 Checking for Android devices..."

    ANDROID_DEVICE=""
    DEVICES_OUTPUT=$(get_flutter_devices)
    if echo "$DEVICES_OUTPUT" | grep -q "android"; then
        # Extract device ID from flutter devices output (format: "name • device_id • platform • details")
        ANDROID_DEVICE=$(echo "$DEVICES_OUTPUT" | grep "android" | head -1 | sed -E 's/.*• ([^ ]+) *• android.*/\1/')
        echo "Found Android device: $ANDROID_DEVICE"
    else
        echo "No Android device found, checking for available emulators..."

        # Get list of Android emulators
        EMULATORS_OUTPUT=$(flutter emulators 2>/dev/null || true)
        ANDROID_EMULATOR=$(echo "$EMULATORS_OUTPUT" | grep "android" | head -1 | awk '{print $1}')

        if [ -n "$ANDROID_EMULATOR" ]; then
            echo "Found Android emulator: $ANDROID_EMULATOR"
            echo "Launching Android emulator..."
            flutter emulators --launch "$ANDROID_EMULATOR" > /dev/null 2>&1 &

            # Wait for emulator to boot (max 2 minutes)
            echo "Waiting for emulator to boot..."
            COUNTER=0
            while [ $COUNTER -lt 120 ]; do
                sleep 2
                DEVICES_OUTPUT=$(get_flutter_devices)
                if echo "$DEVICES_OUTPUT" | grep -q "android"; then
                    # Extract device ID from flutter devices output (format: "name • device_id • platform • details")
                    ANDROID_DEVICE=$(echo "$DEVICES_OUTPUT" | grep "android" | head -1 | sed -E 's/.*• ([^ ]+) *• android.*/\1/')
                    echo "${GREEN}✓ Android emulator ready: $ANDROID_DEVICE${NC}"
                    break
                fi
                COUNTER=$((COUNTER + 2))
            done

            if [ -z "$ANDROID_DEVICE" ]; then
                echo "${RED}✗ Timeout waiting for Android emulator to boot${NC}"
            fi
        else
            echo "${YELLOW}⚠️  No Android emulator found${NC}"
        fi
    fi

    if [ -n "$ANDROID_DEVICE" ]; then
        if run_tests "Android" "-d $ANDROID_DEVICE"; then
            ANDROID_RESULT="✓ PASSED (tests)"
        else
            ANDROID_RESULT="✗ FAILED (tests)"
        fi
    else
        # No device/emulator, fallback to build
        echo "${YELLOW}⚠️  No Android device/emulator available${NC}"
        echo "${BLUE}📦 Building Android APK instead...${NC}"

        if build_platform "Android" "flutter build apk --debug"; then
            ANDROID_RESULT="✓ BUILD OK (APK)"
        else
            ANDROID_RESULT="✗ BUILD FAILED"
        fi
    fi
else
    ANDROID_RESULT="⊘ SKIPPED (not selected)"
fi

# Print summary
echo ""
echo "========================================"
echo "📊 Integration Test Summary"
echo "========================================"
echo ""
echo "macOS:   $MACOS_RESULT"
echo "iOS:     $IOS_RESULT"
echo "Android: $ANDROID_RESULT"
echo ""

# Determine overall result
if [[ "$MACOS_RESULT" == *"FAILED"* ]] || [[ "$IOS_RESULT" == *"FAILED"* ]] || [[ "$ANDROID_RESULT" == *"FAILED"* ]]; then
    echo "${RED}❌ Some tests/builds failed${NC}"
    exit 1
elif [[ "$MACOS_RESULT" == *"SKIPPED"* ]] && [[ "$IOS_RESULT" == *"SKIPPED"* ]] && [[ "$ANDROID_RESULT" == *"SKIPPED"* ]]; then
    echo "${YELLOW}⚠️  All tests were skipped (no platforms available)${NC}"
    exit 1
else
    echo "${GREEN}✅ All available platform tests/builds passed!${NC}"

    # Print build artifact locations
    if [[ "$ANDROID_RESULT" == *"BUILD OK"* ]]; then
        echo ""
        echo "${BLUE}Android APK:${NC} build/app/outputs/flutter-apk/app-debug.apk"
    fi

    if [[ "$IOS_RESULT" == *"BUILD OK"* ]]; then
        echo ""
        echo "${BLUE}iOS App:${NC} build/ios/iphoneos/Runner.app"
    fi

    if [[ "$MACOS_RESULT" == *"BUILD OK"* ]]; then
        echo ""
        echo "${BLUE}macOS App:${NC} build/macos/Build/Products/Debug/executorch_flutter_example.app"
    fi

    echo ""
    exit 0
fi
