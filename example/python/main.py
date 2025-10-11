#!/usr/bin/env python3
"""
ExecuTorch Flutter - Main Script

Unified command-line tool for model export and validation.

Usage:
    python main.py                          # Export models (default)
    python main.py export                   # Export models
    python main.py export --mobilenet       # Export MobileNet only
    python main.py export --yolo yolo11n    # Export YOLO11n
    python main.py validate                 # Validate all models
"""

import sys
import argparse
from pathlib import Path

# Export functions
from validate_all_models import ModelValidator

# Validation functions
import torch
import torchvision.models as models


def export_mobilenet(output_dir="../assets/models"):
    """Export MobileNet V3 Small using Ultralytics-style ExecuTorch export."""
    print("\n" + "="*70)
    print("  Exporting MobileNet V3 Small")
    print("="*70 + "\n")

    try:
        import torch
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from pathlib import Path

        # Load model
        model = models.mobilenet_v3_small(weights='DEFAULT').eval()
        sample_inputs = (torch.randn(1, 3, 224, 224),)

        # Export using official Ultralytics pattern
        et_program = to_edge_transform_and_lower(
            torch.export.export(model, sample_inputs),
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / "mobilenet_v3_small_xnnpack.pte"

        with open(model_file, "wb") as file:
            file.write(et_program.buffer)

        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"✅ MobileNet exported successfully: {model_file.name} ({file_size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolo(model_name="yolo11n", output_dir="../assets/models"):
    """Export YOLO model using Ultralytics-style ExecuTorch export."""
    print("\n" + "="*70)
    print(f"  Exporting {model_name.upper()}")
    print("="*70 + "\n")

    try:
        import torch
        import numpy as np
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from pathlib import Path
        from ultralytics import YOLO

        # Load YOLO model
        model = YOLO(f"{model_name}.pt")

        # Run a dummy prediction to initialize the model (same as export_yolo_official.py)
        np_dummy_tensor = np.ones((640, 640, 3))
        model.predict(np_dummy_tensor, imgsz=(640, 640), device="cpu")

        # Get the PyTorch model and put in eval mode
        pt_model = model.model.cpu().eval()

        # Prepare sample inputs (640x640 for YOLO)
        sample_inputs = (torch.randn(1, 3, 640, 640),)

        # Export using official Ultralytics pattern
        et_program = to_edge_transform_and_lower(
            torch.export.export(pt_model, sample_inputs),
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / f"{model_name}_xnnpack.pte"

        with open(model_file, "wb") as file:
            file.write(et_program.buffer)

        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"✅ {model_name} exported successfully: {model_file.name} ({file_size_mb:.1f} MB)")

        # Clean up downloaded model files (handles both .pt and variant names like yolov5nu.pt)
        for pt_file in Path.cwd().glob("*.pt"):
            if pt_file.stem.startswith(model_name):
                pt_file.unlink()
                print(f"   Cleaned up: {pt_file.name}")

        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_gemma(model_name="gemma-3-270m", output_dir="../assets/models"):
    """Export Gemma text generation model using Optimum ExecuTorch.

    Supported models:
    - gemma-3-270m: 240 MB (text-only, 270M parameters)
    - gemma-3-1b: 892 MB - 1.5 GB (text-only, 1B parameters)

    IMPORTANT: Gemma models are gated on HuggingFace and require:
    1. Request access at https://huggingface.co/google/gemma-3-270m-it
    2. Authenticate with: hf auth login
    3. Install optimum-executorch from source:
       git clone https://github.com/huggingface/optimum-executorch.git
       cd optimum-executorch && pip install '.[dev]' && python install_dev.py
    """
    print("\n" + "="*70)
    print(f"  Exporting {model_name.upper()}")
    print("="*70 + "\n")

    try:
        import subprocess
        from pathlib import Path
        import json
        import shutil

        # Map model name to HuggingFace model ID
        model_id = f"google/{model_name}-it"

        print(f"📦 Exporting {model_name} using Optimum ExecuTorch...")
        print(f"   Model ID: {model_id}")
        print(f"   This may take several minutes on first run...")

        # Create temporary output directory (use absolute path for optimum-cli)
        temp_output = Path(output_dir).resolve() / f"{model_name}_temp"
        temp_output.mkdir(parents=True, exist_ok=True)

        # Use optimum-cli to export the model
        # Note: Removed --use_custom_sdpa and --use_custom_kv_cache flags
        # as they require ExecuTorch features not available in v0.7.0
        cmd = [
            "optimum-cli", "export", "executorch",
            "--model", model_id,
            "--task", "text-generation",
            "--recipe", "xnnpack",
            "--qlinear", "8da4w",
            "--qembedding", "8w",
            "--output_dir", str(temp_output)
        ]

        print(f"\n🔄 Running optimum-cli export...")
        print(f"   Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Export failed with exit code {result.returncode}")
            print(f"   stderr: {result.stderr}")
            return False

        print(f"✅ Model exported successfully!")

        # Find and rename the .pte file
        pte_files = list(temp_output.glob("*.pte"))
        if not pte_files:
            print(f"❌ No .pte file found in {temp_output}")
            return False

        pte_file = pte_files[0]
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        final_pte = output_path / f"{model_name}_xnnpack.pte"

        shutil.move(str(pte_file), str(final_pte))
        file_size_mb = final_pte.stat().st_size / (1024 * 1024)
        print(f"✅ Model file: {final_pte.name} ({file_size_mb:.1f} MB)")

        # Load tokenizer and save vocabulary
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Save simplified vocabulary for Dart
        vocab = tokenizer.get_vocab()
        vocab_file = output_path / f"{model_name}_vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=2)

        print(f"✅ Vocabulary saved: {vocab_file.name} ({len(vocab)} tokens)")

        # Save tokenizer config
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token,
            "model_max_length": 2048,
            "special_tokens": {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            }
        }

        tokenizer_config_file = output_path / f"{model_name}_tokenizer_config.json"
        with open(tokenizer_config_file, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"✅ Tokenizer config saved: {tokenizer_config_file.name}")

        # Clean up temp directory
        shutil.rmtree(temp_output)

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"\n💡 Please install optimum-executorch from source:")
        print(f"   git clone https://github.com/huggingface/optimum-executorch.git")
        print(f"   cd optimum-executorch && pip install '.[dev]' && python install_dev.py")
        return False
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_labels(output_dir="../assets"):
    """Export COCO and ImageNet labels."""
    print("\n" + "="*70)
    print("  Generating Label Files")
    print("="*70 + "\n")

    # COCO labels
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

    print(f"✅ COCO labels: {coco_file}")
    print(f"   ({len(coco_labels)} classes)")


def cmd_export(args):
    """Export command."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ExecuTorch Flutter - Model Export                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    success_count = 0
    total_count = 0

    # Determine what to export
    export_mobilenet_flag = args.all or args.mobilenet
    export_yolo_models = []
    # Note: Gemma is NOT included in --all because it requires HuggingFace authentication
    # Users must explicitly use --gemma flag
    export_gemma_flag = args.gemma

    if args.all:
        export_yolo_models = ["yolo11n", "yolov8n", "yolov5n"]  # All nano YOLO models
    if args.yolo:
        export_yolo_models.extend(args.yolo)

    # Export MobileNet
    if export_mobilenet_flag:
        total_count += 1
        if export_mobilenet(args.output_dir):
            success_count += 1

    # Export YOLO models
    for model_name in export_yolo_models:
        total_count += 1
        if export_yolo(model_name):
            success_count += 1

    # Export Gemma
    if export_gemma_flag:
        total_count += 1
        if export_gemma("gemma-3-270m", args.output_dir):
            success_count += 1

    # Export labels
    if args.all or args.labels:
        export_labels("../assets")

    # Summary
    print("\n" + "="*70)
    print("  Export Summary")
    print("="*70)
    print(f"\n✅ Successfully exported: {success_count}/{total_count} models")

    if success_count < total_count:
        print(f"❌ Failed exports: {total_count - success_count}")

    print(f"\n📁 Models saved to: {args.output_dir}")
    print(f"\n🚀 Next: cd example && flutter run\n")

    return 0 if success_count == total_count else 1


def cmd_validate(args):
    """Validate command."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ExecuTorch Flutter - Model Validation                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Create validator
    validator = ModelValidator(
        models_dir=args.models_dir,
        images_dir=args.images_dir,
        assets_dir="../example/assets"
    )

    # Run validation
    results = validator.validate_all()

    if not results:
        print("\n❌ Validation failed - no results generated")
        return 1

    # Save results
    output_path = Path(args.output_file)
    output_path.write_text(__import__('json').dumps(results, indent=2))

    # Print summary
    print(f"\n{'=' * 70}")
    print("  Validation Summary")
    print(f"{'=' * 70}\n")

    summary = results['summary']
    print(f"✅ Total Models: {summary['total_models_tested']}")
    print(f"   - Classification: {summary['classification_models_count']}")
    print(f"   - Detection: {summary['detection_models_count']}")
    print(f"\n✅ Successful: {summary['successful_models']}")
    print(f"❌ Failed: {summary['failed_models']}")
    print(f"\n📸 Test Images: {summary['total_test_images']}")
    print(f"\n📄 Results: {output_path}\n")

    return 0 if summary['failed_models'] == 0 else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ExecuTorch Flutter - Model Export & Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Export all models (default)
  python main.py export --mobilenet       # Export MobileNet only
  python main.py export --yolo yolo11n    # Export YOLO11n
  python main.py export --gemma           # Export Gemma text generation model
  python main.py export --all             # Export all models
  python main.py validate                 # Validate all models

Supported YOLO models:
  yolo11n, yolo11s, yolov8n, yolov8s, yolov5n, yolov5s

Supported Text Generation models:
  gemma-3-270m (Google Gemma 3, 270M parameters, text-only, ~240 MB)
  gemma-3-1b   (Google Gemma 3, 1B parameters, text-only, ~1.5 GB)

Note: Gemma models require additional setup (advanced):
  1. Install optimum-executorch from source:
     git clone https://github.com/huggingface/optimum-executorch.git
     cd optimum-executorch && pip install '.[dev]' && python install_dev.py
  2. Request access: https://huggingface.co/google/gemma-3-270m-it
  3. Login: hf auth login
  4. Export: python main.py export --gemma
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export models')
    export_parser.add_argument('--all', action='store_true', help='Export all models')
    export_parser.add_argument('--mobilenet', action='store_true', help='Export MobileNet')
    export_parser.add_argument('--yolo', nargs='+', metavar='MODEL', help='Export YOLO model(s)')
    export_parser.add_argument('--gemma', action='store_true', help='Export Gemma text generation model')
    export_parser.add_argument('--labels', action='store_true', help='Generate label files')
    export_parser.add_argument('--output-dir', default='../assets/models',
                                help='Output directory (default: ../assets/models)')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate models')
    validate_parser.add_argument('--models-dir', default='../assets/models',
                                  help='Models directory (default: ../assets/models)')
    validate_parser.add_argument('--images-dir', default='../assets/images',
                                  help='Test images directory (default: ../assets/images)')
    validate_parser.add_argument('--output-file', default='../assets/model_test_results.json',
                                  help='Output file (default: ../assets/model_test_results.json)')

    args = parser.parse_args()

    # Default to export if no command specified
    if args.command is None:
        args.command = 'export'
        args.all = True
        args.mobilenet = False
        args.yolo = None
        args.gemma = False
        args.labels = True
        args.output_dir = '../assets/models'

    # Run command
    if args.command == 'export':
        return cmd_export(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
