#!/bin/bash
# ExecuTorch Installation Script with Optimizations
# This script installs ExecuTorch with XNNPACK backend and all required dependencies

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║        ExecuTorch Installation with Optimizations                ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
echo "📋 Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "❌ Error: Python 3.10 or higher required"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  Step 1: Installing Core Dependencies"
echo "========================================================================"
echo ""

# Install PyTorch first
echo "📦 Installing PyTorch..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "========================================================================"
echo "  Step 2: Installing ExecuTorch"
echo "========================================================================"
echo ""

# Install ExecuTorch with optimizations
echo "📦 Installing ExecuTorch..."
pip3 install executorch

echo ""
echo "========================================================================"
echo "  Step 3: Installing Model Export Dependencies"
echo "========================================================================"
echo ""

# Install model-specific dependencies
echo "📦 Installing Ultralytics (YOLO)..."
pip3 install ultralytics

echo "📦 Installing Transformers (Gemma)..."
pip3 install transformers accelerate

echo "📦 Installing HuggingFace Hub..."
pip3 install huggingface-hub

echo "📦 Installing additional dependencies..."
pip3 install numpy pillow

echo ""
echo "========================================================================"
echo "  Step 4: Installing Optimum-ExecuTorch (for Gemma)"
echo "========================================================================"
echo ""

# Clone and install optimum-executorch
if [ ! -d "/tmp/optimum-executorch" ]; then
    echo "📦 Cloning optimum-executorch..."
    git clone https://github.com/huggingface/optimum-executorch.git /tmp/optimum-executorch
fi

echo "📦 Installing optimum-executorch from source..."
cd /tmp/optimum-executorch
pip3 install '.[dev]'

echo ""
echo "📦 Installing development dependencies (nightly builds)..."
python3 install_dev.py

cd - > /dev/null

echo ""
echo "========================================================================"
echo "  Installation Complete!"
echo "========================================================================"
echo ""
echo "✅ ExecuTorch installed successfully"
echo "✅ All model export dependencies installed"
echo "✅ Optimum-ExecuTorch installed (for Gemma)"
echo ""
echo "Next steps:"
echo "  1. Export models: python3 main.py export --all"
echo "  2. For Gemma: hf auth login (requires HuggingFace access)"
echo "  3. Export Gemma: python3 main.py export --gemma"
echo ""
