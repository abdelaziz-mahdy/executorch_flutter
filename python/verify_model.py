#!/usr/bin/env python3
"""
Verify ExecuTorch model metadata.

Usage:
    python verify_model.py ../example/assets/models/yolo11n_xnnpack.pte
"""

import sys
from pathlib import Path


def verify_model(model_path: str):
    """Verify model metadata."""
    path = Path(model_path)

    if not path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    print(f"📄 Model: {path.name}")
    print(f"📊 Size: {path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        from executorch.runtime import Runtime

        # Load the model
        runtime = Runtime.get()
        program = runtime.load_program(str(path))
        method = program.load_method("forward")

        print("\n✅ Model loaded successfully")
        print(f"\nℹ️  Model appears to be valid ExecuTorch format")
        print(f"   Run in Flutter app to verify input/output shapes")

        return True

    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_model.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    success = verify_model(model_path)
    sys.exit(0 if success else 1)
