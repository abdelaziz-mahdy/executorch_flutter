#!/usr/bin/env python3
"""
Comprehensive Model Validator for ExecuTorch Flutter

This script:
1. Runs all available models (MobileNet, YOLO variants)
2. Tests each model with all test images
3. Records detailed results including:
   - Classification: Top 5 predictions with confidence scores
   - Object Detection: All detected objects with bounding boxes and confidence
4. Saves results to example/assets/model_test_results.json

Usage:
    python validate_all_models.py
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time


class ImageNetPreprocessor:
    """Preprocess images for ImageNet models (MobileNet)."""

    @staticmethod
    def preprocess(image_path: str, target_size: int = 224) -> torch.Tensor:
        """
        Preprocess image for ImageNet models.

        Uses torchvision's standard preprocessing pipeline:
        1. Resize shortest edge to 256 (maintaining aspect ratio)
        2. Center crop to 224x224
        3. Normalize with ImageNet mean/std

        Args:
            image_path: Path to input image
            target_size: Target size (default 224 for MobileNet)

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        from torchvision import transforms

        img = Image.open(image_path).convert('RGB')

        # Use torchvision's standard preprocessing (same as training)
        preprocess = transforms.Compose([
            transforms.Resize(256),  # Resize shorter edge to 256, maintaining aspect ratio
            transforms.CenterCrop(target_size),  # Then center crop to target_size x target_size
            transforms.ToTensor(),  # Convert to [0, 1] and CHW format
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = preprocess(img).unsqueeze(0)
        return img_tensor.float()


class YOLOPreprocessor:
    """Preprocess images for YOLO models."""

    @staticmethod
    def preprocess(image_path: str, target_size: int = 640) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for YOLO models with letterbox.

        Args:
            image_path: Path to input image
            target_size: Target size (default 640 for YOLO)

        Returns:
            Tuple of (preprocessed tensor, metadata)
        """
        img = Image.open(image_path).convert('RGB')
        original_width, original_height = img.size

        # Letterbox resize
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Create canvas with gray padding
        canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(img, (offset_x, offset_y))

        # Convert to tensor
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # Metadata for coordinate transformation
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'target_size': target_size
        }

        return img_tensor.float(), metadata


class YOLOPostprocessor:
    """Postprocess YOLO outputs with NMS."""

    def __init__(self, class_labels: List[str], conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.class_labels = class_labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def postprocess(self, output: np.ndarray, metadata: Dict) -> List[Dict]:
        """
        Postprocess YOLO output with NMS.

        Args:
            output: Raw YOLO output tensor
            metadata: Preprocessing metadata for coordinate transformation

        Returns:
            List of detected objects with bounding boxes
        """
        # Handle different YOLO output formats
        if len(output.shape) == 3:
            # Format: [batch, features, predictions] or [batch, predictions, features]
            if output.shape[1] > output.shape[2]:
                # Transpose format: [batch, features, predictions]
                batch, num_features, num_predictions = output.shape
                is_transposed = True
            else:
                # Normal format: [batch, predictions, features]
                batch, num_predictions, num_features = output.shape
                is_transposed = False
        else:
            print(f"⚠️  Unexpected output shape: {output.shape}")
            return []

        # Detect YOLO version
        is_yolov5 = num_features == 85  # YOLOv5: 4 bbox + 1 objectness + 80 classes
        num_classes = 80 if is_yolov5 else (num_features - 4)

        detections = []

        for i in range(num_predictions):
            # Extract values based on format
            if is_transposed:
                x_center = output[0, 0, i]
                y_center = output[0, 1, i]
                width = output[0, 2, i]
                height = output[0, 3, i]
                objectness = output[0, 4, i] if is_yolov5 else 1.0

                # Get class scores
                class_start = 5 if is_yolov5 else 4
                class_scores = output[0, class_start:class_start + num_classes, i]
            else:
                x_center = output[0, i, 0]
                y_center = output[0, i, 1]
                width = output[0, i, 2]
                height = output[0, i, 3]
                objectness = output[0, i, 4] if is_yolov5 else 1.0

                # Get class scores
                class_start = 5 if is_yolov5 else 4
                class_scores = output[0, i, class_start:class_start + num_classes]

            # Find best class
            best_class_idx = np.argmax(class_scores)
            best_class_conf = class_scores[best_class_idx]

            # Calculate final confidence
            confidence = objectness * best_class_conf if is_yolov5 else best_class_conf

            # Filter by confidence threshold
            if confidence < self.conf_threshold:
                continue

            # Convert to corner coordinates (normalized 0-1)
            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)

            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            class_name = self.class_labels[best_class_idx] if best_class_idx < len(self.class_labels) else f"Class {best_class_idx}"

            detections.append({
                'class': class_name,
                'class_index': int(best_class_idx),
                'confidence': float(confidence),
                'bbox_normalized': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'bbox_pixel': self._to_pixel_coords(
                    x1, y1, x2, y2,
                    metadata['original_width'],
                    metadata['original_height'],
                    metadata['scale'],
                    metadata['offset_x'],
                    metadata['offset_y'],
                    metadata['target_size']
                )
            })

        # Apply NMS
        detections = self._apply_nms(detections)

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def _to_pixel_coords(self, x1, y1, x2, y2, orig_w, orig_h, scale, offset_x, offset_y, target_size):
        """Transform normalized coordinates to original image pixel coordinates."""
        # Convert from normalized to target size
        x1_px = x1 * target_size
        y1_px = y1 * target_size
        x2_px = x2 * target_size
        y2_px = y2 * target_size

        # Remove padding offset
        x1_px = (x1_px - offset_x) / scale
        y1_px = (y1_px - offset_y) / scale
        x2_px = (x2_px - offset_x) / scale
        y2_px = (y2_px - offset_y) / scale

        # Clamp to original image bounds
        x1_px = max(0, min(orig_w, x1_px))
        y1_px = max(0, min(orig_h, y1_px))
        x2_px = max(0, min(orig_w, x2_px))
        y2_px = max(0, min(orig_h, y2_px))

        return {
            'x': int(x1_px),
            'y': int(y1_px),
            'width': int(x2_px - x1_px),
            'height': int(y2_px - y1_px)
        }

    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate IoU between two bounding boxes."""
        b1 = box1['bbox_normalized']
        b2 = box2['bbox_normalized']

        # Intersection
        x_left = max(b1['x1'], b2['x1'])
        y_top = max(b1['y1'], b2['y1'])
        x_right = min(b1['x2'], b2['x2'])
        y_bottom = min(b1['y2'], b2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Areas
        area1 = (b1['x2'] - b1['x1']) * (b1['y2'] - b1['y1'])
        area2 = (b2['x2'] - b2['x1']) * (b2['y2'] - b2['y1'])

        # Union
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        keep = []
        suppressed = [False] * len(detections)

        for i in range(len(detections)):
            if suppressed[i]:
                continue

            keep.append(detections[i])

            # Suppress overlapping boxes of the same class
            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue

                # Only suppress same class
                if detections[i]['class_index'] != detections[j]['class_index']:
                    continue

                iou = self._calculate_iou(detections[i], detections[j])
                if iou > self.iou_threshold:
                    suppressed[j] = True

        return keep


class ModelValidator:
    """Validate all ExecuTorch models."""

    def __init__(self, models_dir: str, images_dir: str, assets_dir: str):
        self.models_dir = Path(models_dir)
        self.images_dir = Path(images_dir)
        self.assets_dir = Path(assets_dir)

        # Load labels
        self.imagenet_labels = self._load_labels(self.assets_dir / "imagenet_classes.txt")
        self.coco_labels = self._load_labels(self.assets_dir / "coco_labels.txt")

        print(f"📋 Loaded {len(self.imagenet_labels)} ImageNet labels")
        print(f"📋 Loaded {len(self.coco_labels)} COCO labels")

    def _load_labels(self, path: Path) -> List[str]:
        """Load class labels from file."""
        if not path.exists():
            print(f"⚠️  Labels not found: {path}")
            return []
        return path.read_text().strip().split('\n')

    def find_models(self) -> Dict[str, List[Path]]:
        """Find all available model files."""
        models = {
            'classification': [],
            'detection': []
        }

        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return models

        for model_file in self.models_dir.glob("*.pte"):
            name = model_file.stem.lower()

            if 'mobilenet' in name or 'resnet' in name or 'efficientnet' in name:
                models['classification'].append(model_file)
            elif 'yolo' in name:
                models['detection'].append(model_file)

        return models

    def find_test_images(self) -> List[Tuple[str, Path]]:
        """Find all test images."""
        images = []

        if not self.images_dir.exists():
            print(f"⚠️  Images directory not found: {self.images_dir}")
            return images

        for img_path in self.images_dir.glob("*.jpg"):
            name = img_path.stem.capitalize()
            images.append((name, img_path))

        images.sort(key=lambda x: x[0])
        return images

    def validate_classification_model(self, model_path: Path, test_images: List[Tuple[str, Path]]) -> Dict:
        """Validate a classification model."""
        print(f"\n{'=' * 70}")
        print(f"  Testing Classification Model: {model_path.name}")
        print(f"{'=' * 70}\n")

        try:
            from executorch.runtime import Runtime

            runtime = Runtime.get()
            program = runtime.load_program(str(model_path))
            method = program.load_method("forward")

            results = {
                'model_name': model_path.stem,
                'model_file': model_path.name,
                'model_type': 'image_classification',
                'input_size': 224,
                'status': 'success',
                'test_results': {}
            }

            preprocessor = ImageNetPreprocessor()

            for img_name, img_path in test_images:
                print(f"  Testing {img_name}...")
                start_time = time.time()

                try:
                    # Preprocess
                    input_tensor = preprocessor.preprocess(str(img_path))

                    # Run inference
                    outputs = method.execute((input_tensor,))
                    inference_time = (time.time() - start_time) * 1000

                    # Get predictions
                    logits = torch.from_numpy(np.array(outputs[0]))
                    probs = torch.nn.functional.softmax(logits.flatten(), dim=-1)
                    top5_prob, top5_idx = torch.topk(probs, 5)

                    # Format results
                    predictions = []
                    for i in range(5):
                        class_idx = top5_idx[i].item()
                        confidence = top5_prob[i].item()
                        class_name = self.imagenet_labels[class_idx] if class_idx < len(self.imagenet_labels) else f"Class {class_idx}"

                        predictions.append({
                            'rank': i + 1,
                            'class': class_name,
                            'class_index': int(class_idx),
                            'confidence': round(confidence, 6),
                            'confidence_percent': f"{confidence * 100:.2f}%"
                        })

                    results['test_results'][img_name] = {
                        'status': 'success',
                        'inference_time_ms': round(inference_time, 2),
                        'top5_predictions': predictions
                    }

                    print(f"    ✅ Top-1: {predictions[0]['class']} ({predictions[0]['confidence_percent']})")
                    print(f"    ⏱️  Inference: {inference_time:.1f}ms")

                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    results['test_results'][img_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            return results

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return {
                'model_name': model_path.stem,
                'model_file': model_path.name,
                'model_type': 'image_classification',
                'status': 'failed',
                'error': str(e)
            }

    def validate_detection_model(self, model_path: Path, test_images: List[Tuple[str, Path]]) -> Dict:
        """Validate an object detection model."""
        print(f"\n{'=' * 70}")
        print(f"  Testing Object Detection Model: {model_path.name}")
        print(f"{'=' * 70}\n")

        try:
            from executorch.runtime import Runtime

            runtime = Runtime.get()
            program = runtime.load_program(str(model_path))
            method = program.load_method("forward")

            results = {
                'model_name': model_path.stem,
                'model_file': model_path.name,
                'model_type': 'object_detection',
                'input_size': 640,
                'status': 'success',
                'test_results': {}
            }

            preprocessor = YOLOPreprocessor()
            postprocessor = YOLOPostprocessor(self.coco_labels, conf_threshold=0.25, iou_threshold=0.45)

            for img_name, img_path in test_images:
                print(f"  Testing {img_name}...")
                start_time = time.time()

                try:
                    # Preprocess
                    input_tensor, metadata = preprocessor.preprocess(str(img_path))

                    # Run inference
                    outputs = method.execute((input_tensor,))
                    inference_time = (time.time() - start_time) * 1000

                    # Postprocess
                    output_array = np.array(outputs[0])
                    detections = postprocessor.postprocess(output_array, metadata)

                    # Format results
                    detection_results = []
                    for det in detections:
                        detection_results.append({
                            'class': det['class'],
                            'class_index': det['class_index'],
                            'confidence': round(det['confidence'], 6),
                            'confidence_percent': f"{det['confidence'] * 100:.2f}%",
                            'bounding_box': det['bbox_pixel']
                        })

                    results['test_results'][img_name] = {
                        'status': 'success',
                        'inference_time_ms': round(inference_time, 2),
                        'num_detections': len(detections),
                        'detections': detection_results
                    }

                    print(f"    ✅ Detected {len(detections)} objects")
                    for det in detections[:3]:  # Show top 3
                        print(f"       - {det['class']}: {det['confidence'] * 100:.1f}%")
                    print(f"    ⏱️  Inference: {inference_time:.1f}ms")

                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results['test_results'][img_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            return results

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return {
                'model_name': model_path.stem,
                'model_file': model_path.name,
                'model_type': 'object_detection',
                'status': 'failed',
                'error': str(e)
            }

    def validate_all(self) -> Dict:
        """Validate all models with all test images."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          ExecuTorch Model Validation - All Models                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

        # Find models and images
        models = self.find_models()
        test_images = self.find_test_images()

        print(f"\n📊 Found {len(models['classification'])} classification models")
        print(f"📊 Found {len(models['detection'])} detection models")
        print(f"📸 Found {len(test_images)} test images\n")

        if not test_images:
            print("❌ No test images found!")
            return {}

        # Validate all models
        all_results = {
            'validation_date': datetime.now().isoformat(),
            'test_images': [name for name, _ in test_images],
            'classification_models': [],
            'detection_models': [],
            'summary': {}
        }

        # Validate classification models
        for model_path in models['classification']:
            result = self.validate_classification_model(model_path, test_images)
            all_results['classification_models'].append(result)

        # Validate detection models
        for model_path in models['detection']:
            result = self.validate_detection_model(model_path, test_images)
            all_results['detection_models'].append(result)

        # Generate summary
        total_models = len(models['classification']) + len(models['detection'])
        successful_models = sum(
            1 for m in all_results['classification_models'] + all_results['detection_models']
            if m['status'] == 'success'
        )

        all_results['summary'] = {
            'total_models_tested': total_models,
            'successful_models': successful_models,
            'failed_models': total_models - successful_models,
            'total_test_images': len(test_images),
            'classification_models_count': len(models['classification']),
            'detection_models_count': len(models['detection'])
        }

        return all_results


def main():
    """Main validation workflow."""
    # Paths
    models_dir = "../example/assets/models"
    images_dir = "../example/assets/images"
    assets_dir = "../example/assets"
    output_file = "../example/assets/model_test_results.json"

    # Create validator
    validator = ModelValidator(models_dir, images_dir, assets_dir)

    # Run validation
    results = validator.validate_all()

    if not results:
        print("\n❌ Validation failed - no results generated")
        return 1

    # Save results
    output_path = Path(output_file)
    output_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'=' * 70}")
    print("  Validation Summary")
    print(f"{'=' * 70}\n")

    summary = results['summary']
    print(f"✅ Total Models Tested: {summary['total_models_tested']}")
    print(f"   - Classification: {summary['classification_models_count']}")
    print(f"   - Detection: {summary['detection_models_count']}")
    print(f"\n✅ Successful: {summary['successful_models']}")
    print(f"❌ Failed: {summary['failed_models']}")
    print(f"\n📸 Test Images: {summary['total_test_images']}")

    print(f"\n{'=' * 70}")
    print(f"📄 Results saved to: {output_path}")
    print(f"{'=' * 70}\n")

    return 0 if summary['failed_models'] == 0 else 1


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
