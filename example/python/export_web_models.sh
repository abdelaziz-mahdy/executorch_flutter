#!/bin/bash
# Export ExecuTorch models for Web/Wasm platform
#
# Web requires the "portable" backend because XNNPACK and other
# hardware-optimized backends contain native code that won't run in WebAssembly.
#
# Usage:
#   ./export_web_models.sh           # Export all models
#   ./export_web_models.sh mobilenet # Export MobileNet only
#   ./export_web_models.sh yolo      # Export all YOLO models only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  ExecuTorch Web Model Export"
echo "========================================"
echo ""

# Check Python dependencies
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch not installed. Run:"
    echo "  pip install torch torchvision executorch ultralytics"
    exit 1
fi

# Determine what to export
EXPORT_TARGET="${1:-all}"

case "$EXPORT_TARGET" in
    all)
        echo "Exporting ALL models with portable backend for web..."
        echo ""
        python3 main.py export --all --backends portable
        ;;
    mobilenet)
        echo "Exporting MobileNet with portable backend for web..."
        echo ""
        python3 main.py export --mobilenet --backends portable
        ;;
    yolo)
        echo "Exporting all YOLO models with portable backend for web..."
        echo ""
        python3 main.py export --yolo yolo11n yolov8n yolov5n --backends portable
        ;;
    *)
        echo "Usage: $0 [all|mobilenet|yolo]"
        echo ""
        echo "Options:"
        echo "  all       - Export MobileNet + all YOLO models (default)"
        echo "  mobilenet - Export MobileNet V3 Small only"
        echo "  yolo      - Export YOLO11n, YOLOv8n, YOLOv5n"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "  Web Models Exported Successfully!"
echo "========================================"
echo ""
echo "Models saved to: ../assets/models/"
echo ""
echo "Web-compatible models:"
ls -lh ../assets/models/*_portable.pte 2>/dev/null || echo "  (no portable models found)"
echo ""
echo "Next steps:"
echo "  1. cd ../example"
echo "  2. flutter run -d chrome"
