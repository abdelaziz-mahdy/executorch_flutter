#!/usr/bin/env python3
"""
Complete Model Setup Script for ExecuTorch Flutter Example App

This script:
1. Checks and installs required Python dependencies
2. Exports MobileNet V3 for image classification
3. Exports YOLO11n for object detection
4. Generates COCO labels file
5. Verifies all assets are ready

Run this script to set up all models for the example app.
"""

import sys
import subprocess
import os
from pathlib import Path


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print("   Current version:", sys.version)
        sys.exit(1)

    print("✅ Python version compatible")


def install_dependencies():
    """Install required Python packages."""
    print_header("Installing Dependencies")

    packages = [
        "torch>=2.1.0",
        "torchvision",
        "executorch",
        "ultralytics",
        "opencv-python",
        "torchao",
    ]

    print("Installing packages:")
    for pkg in packages:
        print(f"  • {pkg}")
    print()

    try:
        # Install all packages at once
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + packages,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\nTry installing manually:")
        for pkg in packages:
            print(f"  pip install {pkg}")
        return False


def verify_imports():
    """Verify all required modules can be imported."""
    print_header("Verifying Imports")

    modules = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("executorch.exir", "ExecuTorch"),
        ("ultralytics", "Ultralytics"),
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - Not available")
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some modules are missing. Run pip install manually.")
        return False

    print("\n✅ All imports verified")
    return True


def export_mobilenet():
    """Export MobileNet V3 model."""
    print_header("Exporting MobileNet V3")

    try:
        # Import after dependencies are installed
        from export_models import export_mobilenet_v3

        export_mobilenet_v3()
        return True
    except Exception as e:
        print(f"❌ MobileNet export failed: {e}")
        return False


def export_yolo():
    """Export YOLO11n model using official ExecuTorch script."""
    print_header("Exporting YOLO11 Nano")

    try:
        # Use official ExecuTorch export script
        script_path = Path(__file__).parent / "export_yolo_official.py"
        if not script_path.exists():
            print(f"❌ Export script not found: {script_path}")
            return False

        print("🚀 Using official ExecuTorch YOLO export script...")
        result = subprocess.run(
            [sys.executable, str(script_path), "--model_name", "yolo11n.pt", "--backend", "xnnpack"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Move exported file to correct location
            exported_file = "yolo11n.pt_fp32_xnnpack.pte"
            if Path(exported_file).exists():
                target_dir = Path(__file__).parent.parent / "example" / "assets" / "models"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / "yolo11n_xnnpack.pte"
                Path(exported_file).rename(target_file)
                size_mb = target_file.stat().st_size / (1024 * 1024)
                print(f"✅ YOLO11n exported successfully!")
                print(f"   Output: {target_file}")
                print(f"   Size: {size_mb:.1f} MB")

                # Clean up downloaded model file
                model_file = Path(__file__).parent / "yolo11n.pt"
                if model_file.exists():
                    model_file.unlink()
                    print(f"✓ Cleaned up intermediate files")

                return True
            else:
                print(f"⚠️  Export completed but file not found: {exported_file}")
                return False
        else:
            print(f"❌ YOLO export failed")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ YOLO export failed: {e}")
        print("\nYou can export YOLO manually later:")
        print("  cd python")
        print("  ./export_yolo_models.sh yolo11n.pt")
        return False


def export_coco_labels():
    """Export COCO labels."""
    print_header("Generating COCO Labels")

    try:
        from export_models import export_coco_labels

        export_coco_labels()
        return True
    except Exception as e:
        print(f"❌ COCO labels export failed: {e}")
        return False


def verify_assets():
    """Verify all expected model files exist."""
    print_header("Verifying Model Assets")

    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "example" / "assets" / "models"
    assets_dir = base_dir / "example" / "assets"

    # Ensure imagenet_classes.txt exists in both locations
    imagenet_src = models_dir / "imagenet_classes.txt"
    imagenet_dst = assets_dir / "imagenet_classes.txt"
    if imagenet_src.exists() and not imagenet_dst.exists():
        import shutil
        shutil.copy(imagenet_src, imagenet_dst)
        print(f"  ℹ️  Copied imagenet_classes.txt to assets root")

    expected_files = {
        "MobileNet V3": models_dir / "mobilenet_v3_small_xnnpack.pte",
        "ImageNet Labels (models)": models_dir / "imagenet_classes.txt",
        "ImageNet Labels (assets)": assets_dir / "imagenet_classes.txt",
        "COCO Labels": assets_dir / "coco_labels.txt",
    }

    optional_files = {
        "YOLO11 Nano": models_dir / "yolo11n_xnnpack.pte",
    }

    all_required_exist = True

    print("Required files:")
    for name, path in expected_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {name}: {path.name} - Missing")
            all_required_exist = False

    print("\nOptional files:")
    for name, path in optional_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  {name}: {path.name} - Not exported (run export_yolo.py)")

    return all_required_exist


def print_summary(success):
    """Print final summary."""
    print("\n" + "=" * 70)
    if success:
        print("  ✅ Setup Complete!")
    else:
        print("  ⚠️  Setup Completed with Warnings")
    print("=" * 70)

    print("\n📱 Next Steps:")
    print("  1. cd example")
    print("  2. flutter pub get")
    print("  3. flutter run")

    if not success:
        print("\n⚠️  Note: Some models may be missing.")
        print("   The app will still run but some features may not work.")
        print("   You can export missing models manually:")
        print("     cd python")
        print("     python3 export_yolo.py")

    print("\n📖 Documentation:")
    print("  • Model Export Guide: MODEL_EXPORT_GUIDE.md")
    print("  • Example App Guide: example/MODEL_EXPORT_GUIDE.md")
    print("  • YOLO Export: python/export_yolo_models.sh")
    print()


def main():
    """Main setup workflow."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║    ExecuTorch Flutter - Model Setup Script                      ║
    ║                                                                  ║
    ║    This script will set up all models for the example app       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Track overall success
    all_success = True

    # Step 1: Check Python version
    check_python_version()

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed")
        print("Please install dependencies manually and run this script again")
        sys.exit(1)

    # Step 3: Verify imports
    if not verify_imports():
        print("\n⚠️  Import verification failed")
        print("Please check your installation and try again")
        sys.exit(1)

    # Step 4: Export MobileNet
    if not export_mobilenet():
        all_success = False

    # Step 5: Export COCO labels
    if not export_coco_labels():
        all_success = False

    # Step 6: Export YOLO using official ExecuTorch script
    yolo_success = export_yolo()
    if not yolo_success:
        print("\n⚠️  YOLO export failed")
        print("   You can export YOLO manually later:")
        print("   cd python && ./export_yolo_models.sh yolo11n.pt")
        print("   The example app will work with MobileNet V3 classification.")

    # Step 7: Verify all assets
    if not verify_assets():
        all_success = False

    # Step 8: Print summary
    print_summary(all_success)

    # Exit with appropriate code
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
