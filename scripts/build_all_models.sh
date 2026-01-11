#!/bin/bash

# Build All Models Script
# Exports all ExecuTorch models directly to the models submodule.
# Models are hosted at: https://github.com/abdelaziz-mahdy/executorch_flutter_models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$REPO_ROOT/example/python"
MODELS_DIR="$REPO_ROOT/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                  ║${NC}"
    echo -e "${CYAN}║        ExecuTorch Flutter - Build All Models                     ║${NC}"
    echo -e "${CYAN}║                                                                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Parse arguments
# Default: Build ALL backends
BACKENDS="xnnpack coreml mps vulkan portable"
INCLUDE_GEMMA=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --backends <list>  Backends to export (default: xnnpack coreml mps vulkan portable)"
    echo "                     Available: xnnpack, coreml, mps, vulkan, portable"
    echo "  --gemma            Include Gemma text generation model (requires HuggingFace auth)"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Export ALL models with all backends"
    echo "  $0 --backends xnnpack        # Export XNNPACK only"
    echo "  $0 --gemma                   # Include Gemma model (requires HuggingFace auth)"
    echo ""
    echo "Note: Models are exported directly to the 'models/' submodule directory."
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --backends)
            BACKENDS="$2"
            shift 2
            ;;
        --gemma)
            INCLUDE_GEMMA=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

print_header

# Check if models directory exists (it's a git repo)
if [ ! -d "$MODELS_DIR/.git" ] && [ ! -f "$MODELS_DIR/.git" ]; then
    echo -e "${YELLOW}Models directory not initialized as git repo.${NC}"
    echo ""
    echo "To initialize:"
    echo -e "  ${CYAN}cd models && git init${NC}"
    echo ""
    echo "Or clone existing repo:"
    echo -e "  ${CYAN}git clone https://github.com/abdelaziz-mahdy/executorch_flutter_models models${NC}"
    echo ""
    exit 1
fi

# Check Python directory exists
if [ ! -d "$PYTHON_DIR" ]; then
    echo -e "${RED}Error: Python directory not found: $PYTHON_DIR${NC}"
    exit 1
fi

# Check if Python requirements are installed
print_section "Checking Dependencies"
cd "$PYTHON_DIR"

if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Warning: PyTorch not installed${NC}"
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

if ! python3 -c "import executorch" 2>/dev/null; then
    echo -e "${YELLOW}Warning: ExecuTorch not installed${NC}"
    echo "Please install ExecuTorch: pip install executorch"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"

# Export models directly to models/ directory
print_section "Exporting Models"

# Build backend argument
BACKEND_ARG=""
if [ -n "$BACKENDS" ]; then
    BACKEND_ARG="--backends $BACKENDS"
fi

# Calculate relative path from python dir to models dir
# From example/python/ to ../../models
OUTPUT_DIR="../../models"

echo -e "${BLUE}Output directory: ${GREEN}$MODELS_DIR${NC}"
echo -e "${BLUE}Backends: ${GREEN}$BACKENDS${NC}"
echo ""

# Export all standard models directly to models/
echo -e "${CYAN}▶ Exporting MobileNet + YOLO models...${NC}"
python3 main.py export --all $BACKEND_ARG --output-dir "$OUTPUT_DIR"

# Export Gemma if requested
if [ "$INCLUDE_GEMMA" = true ]; then
    echo ""
    echo -e "${CYAN}▶ Exporting Gemma model...${NC}"
    echo -e "${YELLOW}Note: Requires HuggingFace authentication${NC}"
    python3 main.py export --gemma --output-dir "$OUTPUT_DIR"
fi

# Summary
print_section "Summary"

# Count and show models
if [ "$(ls -A "$MODELS_DIR"/*.pte 2>/dev/null)" ]; then
    total_size=$(du -sh "$MODELS_DIR" | cut -f1)
    file_count=$(ls "$MODELS_DIR"/*.pte 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  Total: ${GREEN}$file_count models${NC}, ${BLUE}$total_size${NC}"
    echo ""

    # List all models
    echo -e "${CYAN}Models in directory:${NC}"
    for file in "$MODELS_DIR"/*.pte; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            size=$(du -h "$file" | cut -f1)
            echo -e "  • $filename (${BLUE}$size${NC})"
        fi
    done
else
    echo -e "${YELLOW}No .pte files found in $MODELS_DIR${NC}"
fi

# Git instructions
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Next Steps:${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Commit and push models:"
echo -e "     ${CYAN}cd models${NC}"
echo -e "     ${CYAN}git add .${NC}"
echo -e "     ${CYAN}git commit -m \"Update model files\"${NC}"
echo -e "     ${CYAN}git push${NC}"
echo ""

echo -e "${GREEN}✓ Done!${NC}"
echo ""
