#!/bin/bash

# Build All Models Script
# Exports all ExecuTorch models to the models submodule.
# Models are hosted at: https://github.com/abdelaziz-mahdy/executorch_flutter_models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$REPO_ROOT/example/python"
ASSETS_DIR="$REPO_ROOT/example/assets/models"
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
BACKENDS="xnnpack coreml mps"
INCLUDE_WEB=false
INCLUDE_GEMMA=false
SKIP_EXPORT=false
SYNC_ONLY=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --backends <list>  Backends to export (default: xnnpack coreml mps)"
    echo "                     Available: xnnpack, coreml, mps, vulkan, portable"
    echo "  --web              Include web/portable backend models"
    echo "  --gemma            Include Gemma text generation model (requires HuggingFace auth)"
    echo "  --sync-only        Only sync existing models without exporting"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Export all models with default backends"
    echo "  $0 --web                     # Include web/portable models"
    echo "  $0 --backends xnnpack        # Export XNNPACK only"
    echo "  $0 --sync-only               # Only sync existing models to submodule"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --backends)
            BACKENDS="$2"
            shift 2
            ;;
        --web)
            INCLUDE_WEB=true
            shift
            ;;
        --gemma)
            INCLUDE_GEMMA=true
            shift
            ;;
        --sync-only)
            SYNC_ONLY=true
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

# Check if models submodule is initialized
if [ ! -d "$MODELS_DIR/.git" ] && [ ! -f "$MODELS_DIR/.git" ]; then
    echo -e "${YELLOW}Models submodule not initialized.${NC}"
    echo ""
    echo "To initialize the submodule:"
    echo -e "  ${CYAN}git submodule update --init${NC}"
    echo ""
    echo "Or if the repo doesn't exist yet, create it first:"
    echo "  1. Create https://github.com/abdelaziz-mahdy/executorch_flutter_models"
    echo "  2. Run: git submodule update --init"
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

# Export models
if [ "$SYNC_ONLY" = false ]; then
    print_section "Exporting Models"

    # Build backend argument
    BACKEND_ARG=""
    if [ -n "$BACKENDS" ]; then
        BACKEND_ARG="--backends $BACKENDS"
    fi

    echo -e "${BLUE}Backends: ${GREEN}$BACKENDS${NC}"
    echo ""

    # Export all standard models
    echo -e "${CYAN}▶ Exporting MobileNet + YOLO models...${NC}"
    python3 main.py export --all $BACKEND_ARG

    # Export web models if requested
    if [ "$INCLUDE_WEB" = true ]; then
        echo ""
        echo -e "${CYAN}▶ Exporting Web/Portable models...${NC}"
        python3 main.py export --all --backends portable
    fi

    # Export Gemma if requested
    if [ "$INCLUDE_GEMMA" = true ]; then
        echo ""
        echo -e "${CYAN}▶ Exporting Gemma model...${NC}"
        echo -e "${YELLOW}Note: Requires HuggingFace authentication${NC}"
        python3 main.py export --gemma
    fi
fi

# Sync models to submodule
print_section "Syncing Models to Submodule"

# Count files
if [ -d "$ASSETS_DIR" ]; then
    source_count=$(find "$ASSETS_DIR" -name "*.pte" 2>/dev/null | wc -l | tr -d ' ')
else
    source_count=0
fi

if [ "$source_count" -eq 0 ]; then
    echo -e "${YELLOW}No .pte files found in $ASSETS_DIR${NC}"
    echo ""
    echo "Run without --sync-only to export models first."
    exit 0
fi

echo -e "${GREEN}Found $source_count model file(s)${NC}"
echo ""

# Copy files
copied=0
skipped=0
updated=0

for src_file in "$ASSETS_DIR"/*.pte; do
    if [ -f "$src_file" ]; then
        filename=$(basename "$src_file")
        dest_file="$MODELS_DIR/$filename"

        # Check if file already exists and is the same
        if [ -f "$dest_file" ]; then
            src_hash=$(md5 -q "$src_file" 2>/dev/null || md5sum "$src_file" | cut -d' ' -f1)
            dest_hash=$(md5 -q "$dest_file" 2>/dev/null || md5sum "$dest_file" | cut -d' ' -f1)

            if [ "$src_hash" = "$dest_hash" ]; then
                echo -e "  ${YELLOW}⏭${NC}  $filename (unchanged)"
                skipped=$((skipped + 1))
                continue
            else
                echo -e "  ${BLUE}↻${NC}  $filename (updated)"
                updated=$((updated + 1))
            fi
        else
            echo -e "  ${GREEN}+${NC}  $filename (new)"
            copied=$((copied + 1))
        fi

        cp "$src_file" "$dest_file"
    fi
done

# Summary
print_section "Summary"

echo -e "  New files:     ${GREEN}$copied${NC}"
echo -e "  Updated:       ${BLUE}$updated${NC}"
echo -e "  Unchanged:     ${YELLOW}$skipped${NC}"
echo ""

# Show total size
if [ "$(ls -A "$MODELS_DIR"/*.pte 2>/dev/null)" ]; then
    total_size=$(du -sh "$MODELS_DIR" | cut -f1)
    file_count=$(ls "$MODELS_DIR"/*.pte 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  Total: ${GREEN}$file_count models${NC}, ${BLUE}$total_size${NC}"
    echo ""
fi

# List all models
echo -e "${CYAN}Models in submodule:${NC}"
for file in "$MODELS_DIR"/*.pte; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        size=$(du -h "$file" | cut -f1)
        echo -e "  • $filename (${BLUE}$size${NC})"
    fi
done

# Git instructions
if [ $copied -gt 0 ] || [ $updated -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Next Steps:${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  1. Commit and push models submodule:"
    echo -e "     ${CYAN}cd models${NC}"
    echo -e "     ${CYAN}git add .${NC}"
    echo -e "     ${CYAN}git commit -m \"Update model files\"${NC}"
    echo -e "     ${CYAN}git push${NC}"
    echo ""
    echo "  2. Update submodule reference in main repo:"
    echo -e "     ${CYAN}cd ..${NC}"
    echo -e "     ${CYAN}git add models${NC}"
    echo -e "     ${CYAN}git commit -m \"Update models submodule\"${NC}"
    echo -e "     ${CYAN}git push${NC}"
    echo ""
fi

echo -e "${GREEN}✓ Done!${NC}"
echo ""
