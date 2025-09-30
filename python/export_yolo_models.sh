#!/bin/bash
#
# ExecuTorch Flutter - YOLO Model Export Script
#
# This script uses the official ExecuTorch YOLO export script to generate
# YOLO models for the Flutter example app.
#
# Supported Models:
#   - YOLOv8: yolov8n.pt, yolov8s.pt
#   - YOLO11: yolo11n.pt, yolo11s.pt
#   - YOLO12: yolo12n.pt, yolo12s.pt
#
# Usage:
#   ./export_yolo_models.sh [model_name] [--quantize]
#
# Examples:
#   ./export_yolo_models.sh yolo11n.pt
#   ./export_yolo_models.sh yolov8n.pt --quantize
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_NAME="${1:-yolo11n.pt}"
QUANTIZE="${2}"
OUTPUT_DIR="../example/assets/models"
BACKEND="xnnpack"  # Mobile-optimized backend

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          ExecuTorch Flutter - YOLO Model Export (Official)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if export_yolo_official.py exists
if [ ! -f "export_yolo_official.py" ]; then
    echo -e "${RED}❌ Error: export_yolo_official.py not found${NC}"
    echo "   Please run this script from the python/ directory"
    exit 1
fi

# Check Python dependencies
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
python3 -c "import torch; import executorch; import ultralytics; import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Missing dependencies. Installing...${NC}"
    pip install torch torchvision executorch ultralytics opencv-python torchao
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✅ Dependencies OK${NC}"
echo ""

# Print export configuration
echo -e "${BLUE}📋 Export Configuration:${NC}"
echo "   Model:   ${MODEL_NAME}"
echo "   Backend: ${BACKEND}"
echo "   Output:  ${OUTPUT_DIR}"
if [ "$QUANTIZE" == "--quantize" ]; then
    echo "   Mode:    INT8 Quantized"
else
    echo "   Mode:    FP32"
fi
echo ""

# Run export
echo -e "${YELLOW}🚀 Exporting ${MODEL_NAME}...${NC}"
echo ""

if [ "$QUANTIZE" == "--quantize" ]; then
    # Quantized export (requires calibration video)
    echo -e "${YELLOW}⚠️  Quantized export requires a calibration video${NC}"
    echo "   For now, exporting FP32 version..."
    python3 export_yolo_official.py \
        --model_name "${MODEL_NAME}" \
        --backend "${BACKEND}"
else
    # FP32 export
    python3 export_yolo_official.py \
        --model_name "${MODEL_NAME}" \
        --backend "${BACKEND}"
fi

EXPORT_STATUS=$?

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

if [ $EXPORT_STATUS -eq 0 ]; then
    # Find the exported file
    EXPORTED_FILE="${MODEL_NAME}_fp32_${BACKEND}.pte"

    # Extract base name (remove .pt extension)
    MODEL_BASE="${MODEL_NAME%.pt}"
    FINAL_FILE="${MODEL_BASE}_${BACKEND}.pte"

    if [ -f "${EXPORTED_FILE}" ]; then
        # Move and rename to output directory
        mkdir -p "${OUTPUT_DIR}"
        mv "${EXPORTED_FILE}" "${OUTPUT_DIR}/${FINAL_FILE}"

        FILE_SIZE=$(du -h "${OUTPUT_DIR}/${FINAL_FILE}" | cut -f1)

        echo -e "${GREEN}✅ Export successful!${NC}"
        echo ""
        echo "   Output: ${OUTPUT_DIR}/${FINAL_FILE}"
        echo "   Size:   ${FILE_SIZE}"
        echo ""
        echo -e "${GREEN}✓ Model ready for Flutter app${NC}"

        # Clean up downloaded model file (keep only .pte)
        if [ -f "${MODEL_NAME}" ]; then
            rm -f "${MODEL_NAME}"
            echo -e "${GREEN}✓ Cleaned up intermediate files${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Export completed but file not found${NC}"
        echo "   Expected: ${EXPORTED_FILE}"
    fi
else
    echo -e "${RED}❌ Export failed${NC}"
    echo ""
    echo "Possible issues:"
    echo "  • Model requires special preprocessing"
    echo "  • Backend compatibility issues"
    echo "  • Missing dependencies"
    echo ""
    echo "Try:"
    echo "  • Different YOLO model (yolo11n.pt, yolo12n.pt)"
    echo "  • Check error messages above"
    echo "  • Install additional dependencies"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Print next steps
if [ $EXPORT_STATUS -eq 0 ]; then
    echo -e "${BLUE}📱 Next Steps:${NC}"
    echo "  1. Update model config in example/lib/screens/model_playground.dart"
    echo "  2. cd example && flutter run"
    echo "  3. Test object detection with your model"
    echo ""
    echo -e "${BLUE}💡 Export more models:${NC}"
    echo "  ./export_yolo_models.sh yolov8n.pt"
    echo "  ./export_yolo_models.sh yolo11s.pt"
    echo "  ./export_yolo_models.sh yolo12n.pt"
    echo ""
fi

exit $EXPORT_STATUS
