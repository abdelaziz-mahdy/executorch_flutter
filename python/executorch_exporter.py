#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Generic ExecuTorch Model Export Utility
#
# This utility allows external developers to export any PyTorch model
# to ExecuTorch format with automatic backend optimization selection.

import argparse
import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.export import export

# ExecuTorch imports
from executorch.exir import to_edge, EdgeCompileConfig, to_edge_transform_and_lower


@dataclass
class ExportConfig:
    """Configuration for model export."""
    model_name: str
    backends: List[str]
    output_dir: str
    quantize: bool = False
    input_shapes: Optional[List[List[int]]] = None
    input_dtypes: Optional[List[str]] = None
    optimize_for_mobile: bool = True
    enable_dynamic_shape: bool = False
    export_format: str = "pte"  # Future: support other formats


@dataclass
class ExportResult:
    """Result of model export operation."""
    model_name: str
    backend: str
    output_path: str
    file_size_mb: float
    success: bool
    error_message: Optional[str] = None
    export_time_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ExecuTorchExporter:
    """Generic ExecuTorch model exporter with backend auto-detection."""

    # Supported backends with their platform targets
    BACKEND_INFO = {
        "portable": {
            "platforms": ["any"],
            "description": "Generic ExecuTorch runtime (CPU)",
            "requirements": []
        },
        "xnnpack": {
            "platforms": ["android", "ios", "linux", "windows"],
            "description": "Optimized CPU backend",
            "requirements": []
        },
        "coreml": {
            "platforms": ["ios", "macos"],
            "description": "Apple Neural Engine optimization",
            "requirements": ["coremltools"]
        },
        "mps": {
            "platforms": ["ios", "macos"],
            "description": "Metal Performance Shaders (GPU)",
            "requirements": []
        },
        "vulkan": {
            "platforms": ["android", "linux", "windows"],
            "description": "Cross-platform GPU acceleration",
            "requirements": []
        },
        "qnn": {
            "platforms": ["android"],
            "description": "Qualcomm AI Engine (Snapdragon)",
            "requirements": []
        },
        "arm": {
            "platforms": ["embedded", "android"],
            "description": "ARM Ethos-U NPU",
            "requirements": []
        }
    }

    def __init__(self):
        self.available_backends = self._detect_available_backends()

    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which backends are available in the current environment."""
        available = {}

        backend_imports = {
            "coreml": "executorch.backends.apple.coreml.partition.CoreMLPartitioner",
            "xnnpack": "executorch.backends.xnnpack.partition.xnnpack_partitioner.XnnpackPartitioner",
            "vulkan": "executorch.backends.vulkan.partition.vulkan_partitioner.VulkanPartitioner",
            "qnn": "executorch.backends.qualcomm.partition.qnn_partitioner.QnnPartitioner",
            "arm": "executorch.backends.arm.partition.arm_partitioner.ArmPartitioner"
        }

        # Portable and MPS are always available if ExecuTorch is installed
        available["portable"] = True
        available["mps"] = True

        for backend, import_path in backend_imports.items():
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
                available[backend] = True
            except (ImportError, AttributeError):
                available[backend] = False

        return available

    def _get_backend_partitioner(self, backend: str):
        """Get the appropriate partitioner for the specified backend."""
        if not self.available_backends.get(backend, False):
            return None

        try:
            if backend == "coreml":
                from executorch.backends.apple.coreml.partition import CoreMLPartitioner
                return [CoreMLPartitioner()]
            elif backend == "xnnpack":
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
                return [XnnpackPartitioner()]
            elif backend == "vulkan":
                from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
                return [VulkanPartitioner()]
            elif backend == "qnn":
                from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
                return [QnnPartitioner()]
            elif backend == "arm":
                from executorch.backends.arm.partition.arm_partitioner import ArmPartitioner
                return [ArmPartitioner()]
            else:
                return None
        except ImportError as e:
            warnings.warn(f"Backend {backend} partitioner not available: {e}")
            return None

    def _infer_input_specs(self, model: nn.Module, sample_inputs: Tuple) -> Dict[str, Any]:
        """Infer input specifications from the model and sample inputs."""
        specs = []

        for i, input_tensor in enumerate(sample_inputs):
            if isinstance(input_tensor, torch.Tensor):
                specs.append({
                    "index": i,
                    "shape": list(input_tensor.shape),
                    "dtype": str(input_tensor.dtype),
                    "name": f"input_{i}"
                })

        return {"input_specs": specs, "num_inputs": len(specs)}

    def _apply_optimizations(self, model: nn.Module, config: ExportConfig) -> nn.Module:
        """Apply model optimizations based on configuration."""
        optimized_model = model

        if config.quantize:
            print("Applying dynamic quantization...")
            # Apply dynamic quantization for mobile deployment
            optimized_model = torch.quantization.quantize_dynamic(
                optimized_model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )

        if config.optimize_for_mobile:
            # Additional mobile optimizations can be added here
            pass

        return optimized_model

    def export_model(
        self,
        model: nn.Module,
        sample_inputs: Tuple[torch.Tensor, ...],
        config: ExportConfig
    ) -> List[ExportResult]:
        """
        Export a PyTorch model to ExecuTorch format for specified backends.

        Args:
            model: PyTorch model in eval mode
            sample_inputs: Tuple of sample input tensors
            config: Export configuration

        Returns:
            List of export results for each backend
        """
        results = []

        # Ensure model is in eval mode
        model.eval()

        # Apply optimizations
        optimized_model = self._apply_optimizations(model, config)

        # Infer model metadata
        model_metadata = self._infer_input_specs(model, sample_inputs)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        print(f"Exporting model '{config.model_name}' to {len(config.backends)} backends...")
        print(f"Model metadata: {model_metadata}")

        for backend in config.backends:
            print(f"\nExporting for {backend} backend...")

            if not self.available_backends.get(backend, False):
                result = ExportResult(
                    model_name=config.model_name,
                    backend=backend,
                    output_path="",
                    file_size_mb=0.0,
                    success=False,
                    error_message=f"Backend {backend} not available in environment"
                )
                results.append(result)
                continue

            try:
                import time
                start_time = time.time()

                # Export based on backend type
                if backend == "portable":
                    et_program = to_edge(
                        export(optimized_model, sample_inputs),
                    ).to_executorch()

                elif backend == "mps":
                    # MPS typically doesn't require special partitioning
                    et_program = to_edge(
                        export(optimized_model, sample_inputs),
                    ).to_executorch()

                else:
                    # Backend with specific partitioner
                    partitioner = self._get_backend_partitioner(backend)
                    if partitioner is None:
                        raise RuntimeError(f"Failed to get partitioner for {backend}")

                    compile_config = None
                    if backend == "coreml":
                        compile_config = EdgeCompileConfig(_skip_dim_order=True)

                    et_program = to_edge_transform_and_lower(
                        export(optimized_model, sample_inputs),
                        partitioner=partitioner,
                        compile_config=compile_config,
                    ).to_executorch()

                # Generate filename
                suffix = "_quantized" if config.quantize else ""
                filename = f"{config.model_name}_{backend}{suffix}.{config.export_format}"
                output_path = os.path.join(config.output_dir, filename)

                # Write model to file
                with open(output_path, "wb") as f:
                    et_program.write_to_file(f)

                end_time = time.time()
                export_time = end_time - start_time
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

                result = ExportResult(
                    model_name=config.model_name,
                    backend=backend,
                    output_path=output_path,
                    file_size_mb=round(file_size_mb, 2),
                    success=True,
                    export_time_seconds=round(export_time, 2),
                    metadata={
                        **model_metadata,
                        "backend_info": self.BACKEND_INFO.get(backend, {}),
                        "quantized": config.quantize
                    }
                )

                print(f"✓ Exported {backend}: {filename} ({file_size_mb:.1f} MB)")
                results.append(result)

            except Exception as e:
                result = ExportResult(
                    model_name=config.model_name,
                    backend=backend,
                    output_path="",
                    file_size_mb=0.0,
                    success=False,
                    error_message=str(e)
                )
                print(f"✗ Failed {backend}: {e}")
                results.append(result)

        return results

    def get_recommended_backends(self, target_platform: str) -> List[str]:
        """Get recommended backends for a target platform."""
        recommendations = {
            "ios": ["coreml", "mps", "xnnpack", "portable"],
            "android": ["vulkan", "xnnpack", "qnn", "portable"],
            "embedded": ["arm", "xnnpack", "portable"],
            "desktop": ["vulkan", "xnnpack", "portable"],
            "any": ["xnnpack", "portable"]
        }

        recommended = recommendations.get(target_platform, recommendations["any"])

        # Filter by available backends
        return [backend for backend in recommended if self.available_backends.get(backend, False)]

    def create_export_summary(self, results: List[ExportResult], output_path: str):
        """Create a JSON summary of export results."""
        summary = {
            "export_summary": {
                "total_exports": len(results),
                "successful_exports": sum(1 for r in results if r.success),
                "failed_exports": sum(1 for r in results if not r.success),
                "total_size_mb": sum(r.file_size_mb for r in results if r.success)
            },
            "results": [asdict(result) for result in results],
            "available_backends": self.available_backends,
            "backend_info": self.BACKEND_INFO
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Export summary saved to: {output_path}")


def load_model_from_path(model_path: str, model_class=None) -> nn.Module:
    """Load a PyTorch model from various formats."""
    path = Path(model_path)

    if path.suffix == '.pt' or path.suffix == '.pth':
        # PyTorch state dict or full model
        try:
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict):
                if model_class is None:
                    raise ValueError("model_class required for state dict loading")
                instance = model_class()
                instance.load_state_dict(model)
                return instance.eval()
            return model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    elif path.suffix == '.torchscript':
        # TorchScript model
        return torch.jit.load(model_path, map_location='cpu').eval()

    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")


def create_sample_inputs(input_shapes: List[List[int]], input_dtypes: List[str] = None) -> Tuple[torch.Tensor, ...]:
    """Create sample input tensors from shape specifications."""
    if input_dtypes is None:
        input_dtypes = ["float32"] * len(input_shapes)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int64": torch.int64,
        "int32": torch.int32,
        "uint8": torch.uint8
    }

    inputs = []
    for shape, dtype_str in zip(input_shapes, input_dtypes):
        dtype = dtype_map.get(dtype_str, torch.float32)
        tensor = torch.randn(shape, dtype=dtype)
        inputs.append(tensor)

    return tuple(inputs)


def main():
    parser = argparse.ArgumentParser(
        description="Generic ExecuTorch model export utility"
    )
    parser.add_argument("model_path", help="Path to PyTorch model (.pt, .pth, .torchscript)")
    parser.add_argument("--model-name", required=True, help="Name for the exported model")
    parser.add_argument(
        "--input-shapes",
        nargs="+",
        required=True,
        help="Input tensor shapes (e.g., '1,3,224,224' for image input)"
    )
    parser.add_argument(
        "--input-dtypes",
        nargs="+",
        default=None,
        help="Input tensor data types (default: float32 for all)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Backends to export for (default: auto-detect based on platform)"
    )
    parser.add_argument(
        "--target-platform",
        choices=["ios", "android", "embedded", "desktop", "any"],
        default="any",
        help="Target platform for backend recommendations"
    )
    parser.add_argument("--output-dir", default="./exported_models", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--create-summary", action="store_true", help="Create JSON export summary")

    args = parser.parse_args()

    try:
        # Initialize exporter
        exporter = ExecuTorchExporter()

        # Parse input shapes
        input_shapes = []
        for shape_str in args.input_shapes:
            shape = [int(dim) for dim in shape_str.split(',')]
            input_shapes.append(shape)

        # Create sample inputs
        sample_inputs = create_sample_inputs(input_shapes, args.input_dtypes)

        # Load model
        print(f"Loading model from: {args.model_path}")
        model = load_model_from_path(args.model_path)

        # Determine backends
        if args.backends:
            backends = args.backends
        else:
            backends = exporter.get_recommended_backends(args.target_platform)
            print(f"Auto-selected backends for {args.target_platform}: {backends}")

        # Create export configuration
        config = ExportConfig(
            model_name=args.model_name,
            backends=backends,
            output_dir=args.output_dir,
            quantize=args.quantize,
            input_shapes=input_shapes,
            input_dtypes=args.input_dtypes
        )

        # Export model
        results = exporter.export_model(model, sample_inputs, config)

        # Print summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n📊 Export Summary:")
        print(f"✓ Successful: {len(successful)}")
        print(f"✗ Failed: {len(failed)}")

        if successful:
            total_size = sum(r.file_size_mb for r in successful)
            print(f"📁 Total size: {total_size:.1f} MB")
            print("\nSuccessful exports:")
            for result in successful:
                print(f"  • {result.backend}: {result.output_path}")

        if failed:
            print("\nFailed exports:")
            for result in failed:
                print(f"  • {result.backend}: {result.error_message}")

        # Create summary file
        if args.create_summary:
            summary_path = os.path.join(args.output_dir, f"{args.model_name}_export_summary.json")
            exporter.create_export_summary(results, summary_path)

        return 0 if len(successful) > 0 else 1

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())