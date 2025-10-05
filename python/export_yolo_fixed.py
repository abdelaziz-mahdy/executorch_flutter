#!/usr/bin/env python3
"""
Fixed YOLO Model Export Script for ExecuTorch Flutter

This script exports YOLO models to ExecuTorch format with proper 640x640 input.
Fixes the issue where all YOLO models were actually MobileNet (224x224).

Usage:
    python export_yolo_fixed.py yolo11n
    python export_yolo_fixed.py yolov8n
    python export_yolo_fixed.py yolo12n
"""

import torch
from pathlib import Path
import sys


def export_yolo_model(model_name="yolo11n", output_dir="../example/assets/models"):
    """
    Export YOLO model to ExecuTorch with correct 640x640 input size.

    Args:
        model_name: YOLO model variant (yolo11n, yolov8n, yolo12n, etc.)
        output_dir: Output directory for .pte file
    """
    print(f"🎯 Exporting {model_name} to ExecuTorch")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        from executorch.exir import to_edge
        from torch.export import export, ExportedProgram
        from executorch.exir import EdgeCompileConfig

        # Try to import XNNPACK partitioner
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            use_xnnpack = True
            print("✅ XNNPACK backend available")
        except ImportError:
            print("⚠️  XNNPACK backend not available, using portable backend")
            use_xnnpack = False

        # Load YOLO model
        print(f"\n📥 Loading {model_name}.pt...")
        model_file = f"{model_name}.pt"
        yolo = YOLO(model_file)  # Downloads automatically if not present

        # Get PyTorch model
        print("🔧 Preparing model...")
        pt_model = yolo.model.eval()

        # IMPORTANT: Use 640x640 input for YOLO (not 224x224!)
        print("📐 Input size: [1, 3, 640, 640] (YOLO standard)")
        example_input = (torch.randn(1, 3, 640, 640),)

        print("\n🔄 Step 1/4: Exporting with torch.export...")
        try:
            # Export with strict=False to handle dynamic operations
            exported_program = export(
                pt_model,
                example_input,
                strict=False  # Allow dynamic ops that YOLO might use
            )
            print("   ✅ torch.export successful")
        except Exception as e:
            print(f"   ❌ torch.export failed: {e}")
            print("\n💡 Trying alternative export method...")

            # Alternative: Use torch.jit.trace as fallback
            print("   Using torch.jit.trace instead...")
            traced_model = torch.jit.trace(pt_model, example_input[0])
            print("   ✅ torch.jit.trace successful")

            # Now export the traced model
            exported_program = export(traced_model, example_input)
            print("   ✅ Converted to ExportedProgram")

        print("\n🔄 Step 2/4: Converting to Edge IR...")
        edge_config = EdgeCompileConfig(_check_ir_validity=False)  # Skip some checks for YOLO
        edge_program = to_edge(exported_program, compile_config=edge_config)
        print("   ✅ Edge IR conversion successful")

        # Apply XNNPACK optimization if available
        if use_xnnpack:
            print("\n🔄 Step 3/4: Applying XNNPACK optimization...")
            try:
                edge_program = edge_program.to_backend(XnnpackPartitioner())
                backend_suffix = "xnnpack"
                print("   ✅ XNNPACK optimization successful")
            except Exception as e:
                print(f"   ⚠️  XNNPACK optimization failed: {e}")
                print("   Falling back to portable backend")
                backend_suffix = "portable"
        else:
            print("\n🔄 Step 3/4: Using portable backend (no XNNPACK)")
            backend_suffix = "portable"

        print("\n🔄 Step 4/4: Generating ExecuTorch program...")
        executorch_program = edge_program.to_executorch()
        print("   ✅ ExecuTorch program generated")

        # Save to file
        output_file = Path(output_dir) / f"{model_name}_{backend_suffix}.pte"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving to: {output_file}")
        with open(output_file, "wb") as f:
            executorch_program.write_to_file(f)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 70)
        print("✅ EXPORT SUCCESSFUL!")
        print("=" * 70)
        print(f"📄 Output: {output_file}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"⚡ Backend: {backend_suffix.upper()}")
        print("\n📐 Model Specifications:")
        print(f"   Input:  [1, 3, 640, 640] float32 (NCHW, RGB, range [0,1])")
        print(f"   Output: Detection boxes (varies by YOLO version)")
        print("\n⚠️  Note: NMS (Non-Maximum Suppression) must be applied in post-processing")
        print("         The YoloProcessor class handles this automatically.")
        print("=" * 70)

        return True

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("   pip install torch torchvision")
        print("   pip install ultralytics")
        print("   pip install executorch")
        print("   pip install executorch[xnnpack]  # Optional but recommended")
        return False

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("""
═══════════════════════════════════════════════════════════════════════
        Fixed YOLO Export Script for ExecuTorch Flutter
═══════════════════════════════════════════════════════════════════════

This script fixes the issue where YOLO models were exported as 224x224
MobileNet models instead of proper 640x640 YOLO detection models.

Supported models:
  • yolo11n  - YOLO11 Nano (recommended, latest)
  • yolo11s  - YOLO11 Small
  • yolov8n  - YOLOv8 Nano
  • yolov8s  - YOLOv8 Small
  • yolo12n  - YOLO12 Nano (if available)
  • yolov5n  - YOLOv5 Nano

Usage:
  python export_yolo_fixed.py yolo11n
  python export_yolo_fixed.py yolov8n

═══════════════════════════════════════════════════════════════════════
""")

    # Get model name from command line or use default
    if len(sys.argv) > 1:
        model_name = sys.argv[1].replace('.pt', '')  # Remove .pt if provided
    else:
        model_name = "yolo11n"
        print(f"💡 No model specified, using default: {model_name}")
        print("   To specify: python export_yolo_fixed.py yolov8n\n")

    print(f"🚀 Starting export for: {model_name}\n")

    # Export the model
    success = export_yolo_model(model_name)

    if success:
        print("\n✅ All done! Next steps:")
        print("   1. Verify .pte file is in example/assets/models/")
        print("   2. Update model config in model_playground.dart if needed")
        print("   3. Run: cd example && flutter run")
        print()
    else:
        print("\n❌ Export failed. Please check the error messages above.")
        print()


if __name__ == "__main__":
    main()
