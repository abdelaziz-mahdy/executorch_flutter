#!/usr/bin/env python3
"""
Test Model Outputs - Verify model inference and save results

This script tests exported models with the same test images used in Flutter
and saves the results in a text format for verification.

Usage:
    python test_model_outputs.py
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json


def load_imagenet_labels():
    """Load ImageNet class labels."""
    labels_path = Path("../example/assets/imagenet_classes.txt")
    if not labels_path.exists():
        print("⚠️  ImageNet labels not found")
        return []
    return labels_path.read_text().strip().split('\n')


def load_coco_labels():
    """Load COCO class labels."""
    labels_path = Path("../example/assets/coco_labels.txt")
    if not labels_path.exists():
        print("⚠️  COCO labels not found")
        return []
    return labels_path.read_text().strip().split('\n')


def preprocess_image_mobilenet(image_path: str):
    """Preprocess image for MobileNet (224x224)."""
    img = Image.open(image_path).convert('RGB')

    # Center crop and resize to 224x224
    width, height = img.size
    min_dim = min(width, height)

    # Center crop
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))

    # Resize to 224x224
    img = img.resize((224, 224), Image.Resampling.BILINEAR)

    # Convert to tensor [1, 3, 224, 224] normalized to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    # Convert to NCHW format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def preprocess_image_yolo(image_path: str):
    """Preprocess image for YOLO (640x640 with letterbox)."""
    img = Image.open(image_path).convert('RGB')

    # Letterbox resize to 640x640
    width, height = img.size
    scale = min(640 / width, 640 / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Create 640x640 canvas with gray padding
    canvas = Image.new('RGB', (640, 640), (114, 114, 114))
    offset_x = (640 - new_width) // 2
    offset_y = (640 - new_height) // 2
    canvas.paste(img, (offset_x, offset_y))

    # Convert to tensor [1, 3, 640, 640] normalized to [0, 1]
    img_array = np.array(canvas).astype(np.float32) / 255.0

    # Convert to NCHW format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def test_mobilenet(model_path: str, test_images: list, labels: list):
    """Test MobileNet model and save results."""
    print("\n" + "=" * 70)
    print("  Testing MobileNet V3 Small")
    print("=" * 70 + "\n")

    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(str(model_path))
        method = program.load_method("forward")

        results = {}

        for img_name, img_path in test_images:
            print(f"Testing {img_name}...")

            # Preprocess
            input_tensor = preprocess_image_mobilenet(img_path)

            # Run inference
            outputs = method.execute((input_tensor,))

            # Get predictions
            logits = torch.from_numpy(np.array(outputs[0]))
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top5_prob, top5_idx = torch.topk(probs, 5)

            # Store results
            predictions = []
            for i in range(5):
                class_idx = top5_idx[0][i].item()
                confidence = top5_prob[0][i].item()
                class_name = labels[class_idx] if class_idx < len(labels) else f"Class {class_idx}"
                predictions.append({
                    "rank": i + 1,
                    "class": class_name,
                    "confidence": f"{confidence:.4f}",
                    "confidence_percent": f"{confidence * 100:.2f}%"
                })

            results[img_name] = {
                "image": img_name,
                "model": "MobileNet V3 Small",
                "input_size": "224x224",
                "top5_predictions": predictions
            }

            print(f"  Top prediction: {predictions[0]['class']} ({predictions[0]['confidence_percent']})")

        return results

    except Exception as e:
        print(f"❌ MobileNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_yolo(model_path: str, test_images: list, labels: list):
    """Test YOLO model and save results."""
    print("\n" + "=" * 70)
    print("  Testing YOLO11 Nano")
    print("=" * 70 + "\n")

    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(str(model_path))
        method = program.load_method("forward")

        results = {}

        for img_name, img_path in test_images:
            print(f"Testing {img_name}...")

            # Preprocess
            input_tensor = preprocess_image_yolo(img_path)

            # Run inference
            outputs = method.execute((input_tensor,))

            # Parse YOLO output (simplified - no NMS for now)
            output_tensor = torch.from_numpy(np.array(outputs[0]))

            # Store raw output info
            results[img_name] = {
                "image": img_name,
                "model": "YOLO11 Nano",
                "input_size": "640x640",
                "output_shape": list(output_tensor.shape),
                "note": "Full NMS and bbox decoding done in Flutter YoloProcessor"
            }

            print(f"  Output shape: {output_tensor.shape}")

        return results

    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main test workflow."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          Model Output Verification - Test Script                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Test images (same as in Flutter)
    test_images = [
        ("Cat", "../example/assets/images/cat.jpg"),
        ("Dog", "../example/assets/images/dog.jpg"),
        ("Car", "../example/assets/images/car.jpg"),
        ("Person", "../example/assets/images/person.jpg"),
        ("Street", "../example/assets/images/street.jpg"),
    ]

    # Check if images exist
    missing = [name for name, path in test_images if not Path(path).exists()]
    if missing:
        print(f"⚠️  Missing test images: {', '.join(missing)}")
        print("   Please ensure test images are in example/assets/images/")
        return

    # Load labels
    imagenet_labels = load_imagenet_labels()
    coco_labels = load_coco_labels()

    # Test MobileNet
    mobilenet_path = Path("../example/assets/models/mobilenet_v3_small_xnnpack.pte")
    mobilenet_results = {}
    if mobilenet_path.exists():
        mobilenet_results = test_mobilenet(str(mobilenet_path), test_images, imagenet_labels)
    else:
        print("⚠️  MobileNet model not found, skipping")

    # Test YOLO
    yolo_path = Path("../example/assets/models/yolo11n_xnnpack.pte")
    yolo_results = {}
    if yolo_path.exists():
        yolo_results = test_yolo(str(yolo_path), test_images, coco_labels)
    else:
        print("⚠️  YOLO model not found, skipping")

    # Save results
    all_results = {
        "test_date": "2025-10-05",
        "test_images": [name for name, _ in test_images],
        "mobilenet_v3_small": mobilenet_results,
        "yolo11_nano": yolo_results
    }

    output_path = Path("../example/assets/model_test_results.json")
    output_path.write_text(json.dumps(all_results, indent=2))

    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    print(f"\n📄 Results saved to: {output_path}")
    print(f"\n🔍 Summary:")
    print(f"   MobileNet results: {len(mobilenet_results)} images tested")
    print(f"   YOLO results: {len(yolo_results)} images tested")
    print()


if __name__ == "__main__":
    main()
