#!/usr/bin/env python3
"""
YOLO Model Export Script for ExecuTorch Flutter

This script exports YOLO models (YOLOv5, YOLOv8, YOLO11) to ExecuTorch format
using PyTorch's torch.export API (direct method, no ONNX needed).

Supported Models:
  - YOLOv5:  yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
  - YOLOv8:  yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  - YOLO11:  yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Recommended for mobile: Nano (n) or Small (s) variants
"""

import torch
from pathlib import Path


def export_yolo_direct(model_name="yolo11n.pt", output_dir="../example/assets/models"):
    """
    Export YOLO model directly to ExecuTorch using torch.export (no ONNX needed).

    Args:
        model_name: YOLO model to export (e.g., "yolo11n.pt", "yolov8n.pt", "yolov5n.pt")
        output_dir: Output directory for .pte file
    """
    print(f"🎯 Exporting {model_name} to ExecuTorch (Direct Method)")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        from executorch.exir import to_edge

        # Try to import XNNPACK partitioner for mobile optimization
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            use_xnnpack = True
        except ImportError:
            print("⚠️  XNNPACK backend not available, using portable backend")
            use_xnnpack = False

        print(f"📥 Loading {model_name}...")
        yolo = YOLO(model_name)  # Downloads automatically if not present

        # Get the PyTorch model without NMS (required for ExecuTorch)
        model = yolo.model.eval().cpu()

        print("🔄 Exporting to torch.export format...")
        # Create example input (YOLO standard: 640x640)
        example_input = (torch.zeros(1, 3, 640, 640),)

        # Export to torch.export
        exported_program = torch.export.export(model, example_input)

        print("🔧 Converting to Edge IR...")
        edge_program = to_edge(exported_program)

        # Apply XNNPACK optimization if available
        if use_xnnpack:
            print("⚡ Applying XNNPACK optimization for mobile...")
            edge_program = edge_program.to_backend(XnnpackPartitioner())
            backend_suffix = "xnnpack"
        else:
            backend_suffix = "portable"

        print("📦 Generating ExecuTorch program...")
        executorch_program = edge_program.to_executorch()

        # Create output filename
        model_base = model_name.replace('.pt', '')
        output_file = Path(output_dir) / f"{model_base}_{backend_suffix}.pte"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_file, "wb") as f:
            executorch_program.write_to_file(f)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 70)
        print(f"✅ Successfully exported!")
        print(f"   Output: {output_file}")
        print(f"   Size: {file_size_mb:.1f} MB")
        print(f"   Backend: {backend_suffix.upper()}")
        print("=" * 70)
        print("\n📝 Model Info:")
        print(f"   Input:  [1, 3, 640, 640] (NCHW format, RGB, normalized [0,1])")
        print(f"   Output: [1, 84, 8400] (80 classes + 4 bbox coords)")
        print(f"   Note:   NMS must be applied in post-processing")
        print()

        return True

    except ImportError as e:
        print(f"\n❌ Error: Missing dependency - {e}")
        print("\nInstall required packages:")
        print("  pip install torch")
        print("  pip install ultralytics")
        print("  pip install executorch")
        print()
        return False

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("\n⚠️  YOLO models have known compatibility issues with torch.export")
        print("   due to dynamic operations (e.g., .item(), dynamic shapes).")
        print("\nWorkarounds:")
        print("  1. Export via ONNX format (see export_yolo_via_onnx)")
        print("  2. Use pre-converted YOLO models")
        print("  3. Simplify model by removing dynamic operations")
        print("\nFor now, you can:")
        print("  • Continue with MobileNet V3 for image classification")
        print("  • Export YOLO manually using ONNX workflow")
        print("  • Check for ExecuTorch updates with better YOLO support")
        print()
        return False


def export_yolo_via_onnx(model_name="yolo11n.pt", output_dir="../example/assets/models"):
    """
    Export YOLO to ExecuTorch via ONNX (fallback method).

    This is a two-step process:
    1. Export YOLO to ONNX
    2. Convert ONNX to ExecuTorch
    """
    print(f"🎯 Exporting {model_name} via ONNX (Fallback Method)")
    print("=" * 70)

    try:
        from ultralytics import YOLO

        print(f"📥 Loading {model_name}...")
        model = YOLO(model_name)

        print("🔄 Exporting to ONNX...")
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            dynamic=False  # ExecuTorch requires static shapes
        )

        print(f"✅ ONNX export successful: {onnx_path}")
        print("\n📝 Next Step: Convert ONNX to ExecuTorch")
        print("   Unfortunately, automatic ONNX→ExecuTorch conversion is not")
        print("   currently available in this script.")
        print("\n   Manual conversion:")
        print("   1. Follow: https://pytorch.org/executorch/stable/tutorial-onnx-to-executorch.html")
        print("   2. Or use direct export method (recommended)")
        print()

        return onnx_path

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None


def print_usage_guide():
    """Print comprehensive usage guide."""
    print("""
═══════════════════════════════════════════════════════════════════════
                YOLO + ExecuTorch: Supported Versions
═══════════════════════════════════════════════════════════════════════

This script supports exporting the following YOLO models:

YOLOv5 (Ultralytics):
  • yolov5n.pt - Nano   (~4MB)  - Fastest, good for simple scenes
  • yolov5s.pt - Small  (~14MB) - Balanced speed and accuracy
  • yolov5m.pt - Medium (~40MB) - More accurate
  • yolov5l.pt - Large  (~90MB) - High accuracy (may be slow)
  • yolov5x.pt - XLarge (~170MB)- Best accuracy (not for mobile)

YOLOv8 (Ultralytics):
  • yolov8n.pt - Nano   (~6MB)  - Fastest, improved over v5
  • yolov8s.pt - Small  (~22MB) - Best balance for mobile
  • yolov8m.pt - Medium (~52MB) - High accuracy
  • yolov8l.pt - Large  (~88MB) - Very high accuracy
  • yolov8x.pt - XLarge (~138MB)- Best accuracy (not for mobile)

YOLO11 (Latest, Ultralytics):
  • yolo11n.pt - Nano   (~6MB)  - Fastest, best efficiency
  • yolo11s.pt - Small  (~22MB) - Recommended for mobile
  • yolo11m.pt - Medium (~52MB) - High accuracy
  • yolo11l.pt - Large  (~88MB) - Very high accuracy
  • yolo11x.pt - XLarge (~138MB)- Best accuracy (not for mobile)

Recommendation: Use Nano or Small variants for mobile devices.
                YOLO11 offers the best accuracy/speed tradeoff.

═══════════════════════════════════════════════════════════════════════
                        Export Methods
═══════════════════════════════════════════════════════════════════════

Method 1: Direct Export (Recommended) ⭐
────────────────────────────────────────
Uses torch.export directly, no ONNX intermediate step needed.

  from export_yolo import export_yolo_direct
  export_yolo_direct("yolo11n.pt")

Method 2: Via ONNX (Fallback)
──────────────────────────────
Two-step process: YOLO → ONNX → ExecuTorch

  from export_yolo import export_yolo_via_onnx
  export_yolo_via_onnx("yolo11n.pt")

═══════════════════════════════════════════════════════════════════════
                    Important Technical Details
═══════════════════════════════════════════════════════════════════════

⚠️  NMS (Non-Maximum Suppression)
   Exported YOLO models do NOT include NMS. You must implement NMS in
   post-processing. The YoloProcessor class already handles this.

⚠️  Static Input Size
   ExecuTorch requires fixed 640x640 input. If you need different sizes,
   modify the export script's example_input dimensions.

⚠️  XNNPACK Backend
   XNNPACK provides significant speedup on mobile CPUs. Install with:
   pip install executorch[xnnpack]

📊 Expected Performance (on mobile):
   • yolo11n/yolov8n: 50-100ms per frame (suitable for real-time)
   • yolo11s/yolov8s: 100-200ms per frame (good for video)
   • Larger models: 200ms+ (still images only)

📝 Model Input/Output Format:
   Input:  [1, 3, 640, 640] - NCHW, RGB, float32, range [0,1]
   Output: [1, 84, 8400] - 8400 predictions, each with 4 bbox + 80 classes

   Bbox format: [x_center, y_center, width, height] in image coordinates

═══════════════════════════════════════════════════════════════════════
                        Usage in Flutter
═══════════════════════════════════════════════════════════════════════

1. Place exported .pte file in: example/assets/models/

2. Use YoloProcessor in your Flutter app:

   final processor = YoloProcessor(
     preprocessConfig: YoloPreprocessConfig(
       targetWidth: 640,
       targetHeight: 640,
     ),
     classLabels: cocoLabels,  // 80 COCO classes
     confidenceThreshold: 0.25,
     iouThreshold: 0.45,
   );

   final result = await processor.process(imageBytes, model);

3. COCO labels are automatically created by: python export_models.py

═══════════════════════════════════════════════════════════════════════

For more information:
  • ExecuTorch: https://pytorch.org/executorch/
  • Ultralytics: https://docs.ultralytics.com/
  • Flutter Guide: ../example/MODEL_EXPORT_GUIDE.md

═══════════════════════════════════════════════════════════════════════
""")


def main():
    """Main export workflow."""
    import sys

    print_usage_guide()

    # Check for command line arguments
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        # Default to YOLO11 Nano (latest and most efficient)
        model_name = "yolo11n.pt"
        print(f"\n💡 No model specified, using default: {model_name}")
        print("   To specify a model: python export_yolo.py yolov8n.pt")

    print(f"\n🚀 Starting export for: {model_name}")
    print()

    # Try direct export first (recommended)
    success = export_yolo_direct(model_name)

    if not success:
        print("\n💡 Attempting fallback method (ONNX)...")
        export_yolo_via_onnx(model_name)

    print("\n" + "=" * 70)
    print("📋 Next Steps:")
    print("  1. Verify .pte file exists in example/assets/models/")
    print("  2. Ensure COCO labels exist: python export_models.py")
    print("  3. Update model config in example/lib/screens/model_playground.dart")
    print("  4. Run Flutter app: cd example && flutter run")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
