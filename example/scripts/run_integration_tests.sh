#!/bin/bash

# Integration Test Runner for ExecuTorch Flutter Example App
# This script runs integration tests on all supported platforms

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🧪 ExecuTorch Flutter Integration Tests"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if models exist
echo "📦 Checking for model files..."
MODELS_DIR="$PROJECT_DIR/assets/models"
REQUIRED_MODELS=(
    "mobilenet_v3_small_xnnpack.pte"
    "yolo11n_xnnpack.pte"
    "yolov5n_xnnpack.pte"
    "yolov8n_xnnpack.pte"
)

MODELS_MISSING=false
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$MODELS_DIR/$model" ]; then
        echo "${RED}✗ Missing model: $model${NC}"
        MODELS_MISSING=true
    else
        echo "${GREEN}✓ Found model: $model${NC}"
    fi
done

if [ "$MODELS_MISSING" = true ]; then
    echo ""
    echo "${RED}Error: Some required models are missing!${NC}"
    echo "Please run the model setup script first:"
    echo "  cd ../python && python3 setup_models.py"
    exit 1
fi

echo ""
echo "✅ All required models found"
echo ""

# Test on macOS (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📍 Detected macOS platform"
    if run_tests "macOS" "-d macos"; then
        MACOS_RESULT="✓ PASSED"
    else
        MACOS_RESULT="✗ FAILED"
    fi
else
    echo "⚠️  Skipping macOS tests (not on macOS)"
    MACOS_RESULT="⊘ SKIPPED"
fi

# Test on iOS (launch simulator or use physical device)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "📍 Checking for iOS devices..."

    IOS_DEVICE=""
    # Check if any iOS device is connected
    if flutter devices | grep -q "ios"; then
        IOS_DEVICE=$(flutter devices | grep "ios" | head -1 | awk '{print $5}' | tr -d '•')
        echo "Found iOS device: $IOS_DEVICE"
    else
        echo "No iOS device found, checking for iOS Simulator..."

        # Check for iOS Simulator
        IOS_SIMULATOR=$(flutter emulators | grep "ios" | head -1 | awk '{print $1}')

        if [ -n "$IOS_SIMULATOR" ]; then
            echo "Found iOS Simulator: $IOS_SIMULATOR"
            echo "${YELLOW}⚠️  Note: iOS Simulator (x86_64) is NOT supported for ExecuTorch${NC}"
            echo "   Checking if physical device is available..."
        fi

        echo "${YELLOW}⚠️  No iOS physical device connected${NC}"
        echo "   To run iOS tests, connect a physical iOS device (arm64)"
        echo "   iOS Simulator is NOT supported for ExecuTorch"
    fi

    if [ -n "$IOS_DEVICE" ]; then
        if run_tests "iOS" "-d $IOS_DEVICE"; then
            IOS_RESULT="✓ PASSED"
        else
            IOS_RESULT="✗ FAILED"
        fi
    else
        IOS_RESULT="⊘ SKIPPED (no device)"
    fi
else
    echo "⚠️  Skipping iOS tests (not on macOS)"
    IOS_RESULT="⊘ SKIPPED (not macOS)"
fi

# Test on Android (launch emulator if needed)
echo ""
echo "📍 Checking for Android devices..."

ANDROID_DEVICE=""
if flutter devices | grep -q "android"; then
    ANDROID_DEVICE=$(flutter devices | grep "android" | head -1 | awk '{print $5}' | tr -d '•')
    echo "Found Android device: $ANDROID_DEVICE"
else
    echo "No Android device found, checking for available emulators..."

    # Get list of Android emulators
    ANDROID_EMULATOR=$(flutter emulators | grep "android" | head -1 | awk '{print $1}')

    if [ -n "$ANDROID_EMULATOR" ]; then
        echo "Found Android emulator: $ANDROID_EMULATOR"
        echo "Launching Android emulator..."
        flutter emulators --launch "$ANDROID_EMULATOR" > /dev/null 2>&1 &

        # Wait for emulator to boot (max 2 minutes)
        echo "Waiting for emulator to boot..."
        COUNTER=0
        while [ $COUNTER -lt 120 ]; do
            sleep 2
            if flutter devices | grep -q "android"; then
                ANDROID_DEVICE=$(flutter devices | grep "android" | head -1 | awk '{print $5}' | tr -d '•')
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
        ANDROID_RESULT="✓ PASSED"
    else
        ANDROID_RESULT="✗ FAILED"
    fi
else
    echo "${YELLOW}⚠️  No Android device/emulator available, skipping Android tests${NC}"
    ANDROID_RESULT="⊘ SKIPPED (no device)"
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
    echo "${RED}❌ Some tests failed${NC}"
    exit 1
elif [[ "$MACOS_RESULT" == *"SKIPPED"* ]] && [[ "$IOS_RESULT" == *"SKIPPED"* ]] && [[ "$ANDROID_RESULT" == *"SKIPPED"* ]]; then
    echo "${YELLOW}⚠️  All tests were skipped (no devices available)${NC}"
    exit 1
else
    echo "${GREEN}✅ All available platform tests passed!${NC}"
    exit 0
fi
