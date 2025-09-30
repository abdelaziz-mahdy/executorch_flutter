#!/usr/bin/env python3
"""
Unified Model Export Script for ExecuTorch Flutter

This script exports models used in the Flutter example app.
"""

import torch
import torchvision.models as models
from pathlib import Path
from executorch_exporter import ExecuTorchExporter, ExportConfig


def export_mobilenet_v3():
    """Export MobileNet V3 Small for image classification."""
    print("📱 Exporting MobileNet V3 Small")
    print("=" * 60)

    # Load pretrained model
    model = models.mobilenet_v3_small(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    # Export with XNNPACK backend for mobile
    exporter = ExecuTorchExporter()
    config = ExportConfig(
        model_name="mobilenet_v3_small",
        backends=["xnnpack"],
        output_dir="../example/assets/models",
        quantize=False
    )

    results = exporter.export_model(model, sample_inputs, config)

    successful = [r for r in results if r.success]
    if successful:
        print(f"✅ Successfully exported to: {successful[0].output_path}")
    else:
        print("❌ Export failed")

    print()


def export_yolo_guide():
    """Print YOLO export instructions."""
    print("🎯 YOLO Model Export Instructions")
    print("=" * 60)
    print("""
YOLO models can be exported directly to ExecuTorch (no ONNX needed):

Supported: YOLOv5, YOLOv8, YOLO11 (all variants: n, s, m, l, x)

1. Install dependencies:
   pip install torch ultralytics executorch

2. Export YOLO directly to ExecuTorch:
   python3 export_yolo.py

   Or specify a specific model:
   python3 export_yolo.py yolo11n.pt
   python3 export_yolo.py yolov8n.pt
   python3 export_yolo.py yolov5s.pt

3. The script will:
   - Download the YOLO model automatically (if needed)
   - Export to ExecuTorch using torch.export (direct, no ONNX)
   - Apply XNNPACK optimization for mobile
   - Save as .pte file in example/assets/models/

Recommended models for mobile:
  • yolo11n.pt - Best efficiency (latest)
  • yolov8n.pt - Excellent balance
  • yolov5n.pt - Lightweight and fast

For detailed instructions, see: python/export_yolo.py
""")
    print()


def export_coco_labels():
    """Export COCO class labels."""
    print("🏷️  Exporting COCO Labels")
    print("=" * 60)

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

    output_path = Path("../example/assets/coco_labels.txt")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text('\n'.join(coco_labels))

    print(f"✅ Saved {len(coco_labels)} COCO labels")
    print()


def main():
    """Export models for the Flutter example app."""
    print("🚀 ExecuTorch Flutter Model Export")
    print("=" * 60)
    print()

    print("This script exports models for the Flutter example app.")
    print("Exported models will be placed in: example/assets/models/")
    print()

    # Export classification model
    export_mobilenet_v3()

    # Export COCO labels for object detection
    export_coco_labels()

    # Show YOLO instructions
    export_yolo_guide()

    print("=" * 60)
    print("✅ Export completed!")
    print()
    print("Next steps:")
    print("  1. Ensure models are in example/assets/models/")
    print("  2. Run: cd example && flutter pub get")
    print("  3. Run: flutter run")
    print()


if __name__ == "__main__":
    main()
