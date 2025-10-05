#!/usr/bin/env python3
"""
Unified Model Export Script for ExecuTorch Flutter

This script exports all models needed for the Flutter example app:
- MobileNet V3 Small for image classification (224x224)
- YOLO models for object detection (640x640)

Usage:
    python export_unified.py --all                    # Export all models
    python export_unified.py --mobilenet              # Export MobileNet only
    python export_unified.py --yolo yolo11n           # Export specific YOLO
    python export_unified.py --yolo yolo11n yolov8n   # Export multiple YOLOs

Author: ExecuTorch Flutter Plugin Team
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import List, Optional


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def export_mobilenet(output_dir: str = "../example/assets/models") -> bool:
    """
    Export MobileNet V3 Small for image classification.

    Input: [1, 3, 224, 224] - RGB images, normalized [0,1]
    Output: [1, 1000] - ImageNet class probabilities
    """
    print_section("Exporting MobileNet V3 Small")

    try:
        import torchvision.models as models
        from executorch.exir import to_edge
        from torch.export import export

        # Try XNNPACK backend
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            use_xnnpack = True
            print("✅ XNNPACK backend available")
        except ImportError:
            print("⚠️  XNNPACK not available, using portable backend")
            use_xnnpack = False

        print("📥 Loading MobileNet V3 Small (pretrained)...")
        model = models.mobilenet_v3_small(weights='DEFAULT').eval()

        print("📐 Input size: [1, 3, 224, 224] (ImageNet standard)")
        example_input = (torch.randn(1, 3, 224, 224),)

        print("🔄 Step 1/4: Exporting with torch.export...")
        exported_program = export(model, example_input)
        print("   ✅ torch.export successful")

        print("🔄 Step 2/4: Converting to Edge IR...")
        edge_program = to_edge(exported_program)
        print("   ✅ Edge IR conversion successful")

        if use_xnnpack:
            print("🔄 Step 3/4: Applying XNNPACK optimization...")
            try:
                edge_program = edge_program.to_backend(XnnpackPartitioner())
                backend_suffix = "xnnpack"
                print("   ✅ XNNPACK optimization successful")
            except Exception as e:
                print(f"   ⚠️  XNNPACK failed: {e}")
                print("   Falling back to portable backend")
                backend_suffix = "portable"
        else:
            print("🔄 Step 3/4: Using portable backend...")
            backend_suffix = "portable"

        print("🔄 Step 4/4: Generating ExecuTorch program...")
        executorch_program = edge_program.to_executorch()
        print("   ✅ ExecuTorch program generated")

        # Save to file
        output_file = Path(output_dir) / f"mobilenet_v3_small_{backend_suffix}.pte"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving to: {output_file}")
        with open(output_file, "wb") as f:
            executorch_program.write_to_file(f)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"\n{'=' * 70}")
        print("✅ MOBILENET EXPORT SUCCESSFUL!")
        print(f"{'=' * 70}")
        print(f"📄 Output: {output_file}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"⚡ Backend: {backend_suffix.upper()}")
        print(f"\n📐 Model Specifications:")
        print(f"   Input:  [1, 3, 224, 224] float32 (NCHW, RGB, range [0,1])")
        print(f"   Output: [1, 1000] float32 (ImageNet class probabilities)")
        print(f"{'=' * 70}\n")

        return True

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("   pip install torch torchvision executorch")
        return False
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolo(model_name: str = "yolo11n", output_dir: str = "../example/assets/models") -> bool:
    """
    Export YOLO model for object detection with correct 640x640 input.

    Supported: yolo11n, yolo11s, yolov8n, yolov8s, yolov5n, yolov5s

    Input: [1, 3, 640, 640] - RGB images, normalized [0,1]
    Output: [1, 84, 8400] - Detection boxes (4 bbox + 80 classes)
    """
    print_section(f"Exporting {model_name.upper()}")

    try:
        from ultralytics import YOLO
        from executorch.exir import to_edge, EdgeCompileConfig
        from torch.export import export

        # Try XNNPACK backend
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            use_xnnpack = True
            print("✅ XNNPACK backend available")
        except ImportError:
            print("⚠️  XNNPACK not available, using portable backend")
            use_xnnpack = False

        # Load YOLO model
        print(f"📥 Loading {model_name}.pt...")
        model_file = f"{model_name}.pt"
        yolo = YOLO(model_file)  # Downloads automatically if not present

        # Get PyTorch model
        print("🔧 Preparing model...")
        pt_model = yolo.model.eval()

        # CRITICAL: Use 640x640 input for YOLO (not 224x224!)
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
            print(f"   ⚠️  torch.export with strict=False failed: {e}")
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

        print(f"\n{'=' * 70}")
        print(f"✅ {model_name.upper()} EXPORT SUCCESSFUL!")
        print(f"{'=' * 70}")
        print(f"📄 Output: {output_file}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"⚡ Backend: {backend_suffix.upper()}")
        print(f"\n📐 Model Specifications:")
        print(f"   Input:  [1, 3, 640, 640] float32 (NCHW, RGB, range [0,1])")
        print(f"   Output: Detection boxes (varies by YOLO version)")
        print(f"\n⚠️  Note: NMS (Non-Maximum Suppression) must be applied in post-processing")
        print(f"         The YoloProcessor class handles this automatically.")
        print(f"{'=' * 70}\n")

        return True

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("   pip install torch ultralytics executorch")
        return False
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_labels(output_dir: str = "../example/assets"):
    """Export COCO and ImageNet label files."""
    print_section("Generating Label Files")

    # COCO labels (80 classes)
    coco_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    coco_file = output_path / "coco_labels.txt"
    coco_file.write_text('\n'.join(coco_labels))
    print(f"✅ Saved {len(coco_labels)} COCO labels to: {coco_file}")

    # Note about ImageNet labels
    print(f"\nℹ️  ImageNet labels (imagenet_classes.txt) should already exist")
    print(f"   If missing, download from torchvision or copy from models directory")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified model export script for ExecuTorch Flutter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_unified.py --all                    # Export all models
  python export_unified.py --mobilenet              # Export MobileNet only
  python export_unified.py --yolo yolo11n           # Export YOLO11n
  python export_unified.py --yolo yolo11n yolov8n   # Export multiple YOLOs
  python export_unified.py --labels                 # Generate label files only

Supported YOLO models:
  yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
        """
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all default models (MobileNet + YOLO11n)"
    )
    parser.add_argument(
        "--mobilenet",
        action="store_true",
        help="Export MobileNet V3 Small"
    )
    parser.add_argument(
        "--yolo",
        nargs="+",
        metavar="MODEL",
        help="Export YOLO model(s) (e.g., yolo11n yolov8n)"
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Generate label files (COCO, ImageNet)"
    )
    parser.add_argument(
        "--output-dir",
        default="../example/assets/models",
        help="Output directory for models (default: ../example/assets/models)"
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not (args.all or args.mobilenet or args.yolo or args.labels):
        parser.print_help()
        return 0

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ExecuTorch Flutter - Unified Model Export Script          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    success_count = 0
    total_count = 0

    # Export MobileNet
    if args.all or args.mobilenet:
        total_count += 1
        if export_mobilenet(args.output_dir):
            success_count += 1

    # Export YOLO models
    yolo_models = []
    if args.all:
        yolo_models = ["yolo11n"]  # Default YOLO
    if args.yolo:
        yolo_models.extend(args.yolo)

    for model_name in yolo_models:
        total_count += 1
        if export_yolo(model_name, args.output_dir):
            success_count += 1

    # Generate labels
    if args.all or args.labels:
        export_labels("../example/assets")

    # Summary
    print_section("Export Summary")
    print(f"✅ Successfully exported: {success_count}/{total_count} models")

    if success_count < total_count:
        print(f"⚠️  Failed exports: {total_count - success_count}")

    print(f"\n📁 Models saved to: {args.output_dir}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Verify .pte files in {args.output_dir}/")
    print(f"   2. Run: cd example && flutter run")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
