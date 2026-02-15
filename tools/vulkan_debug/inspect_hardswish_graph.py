#!/usr/bin/env python3
"""Inspect the full computation graph for a Hardswish model exported via Vulkan.

Shows every stage: ATen -> Edge IR -> Vulkan preprocess (with prepack nodes) -> VkGraph.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from functools import partial


class HardswishModel(nn.Module):
    def forward(self, x):
        return torch.nn.functional.hardswish(x)


def print_graph_nodes(gm, title, show_constants=True):
    """Print all nodes in a graph module with full details."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    print(f"\n  Graph text:")
    for line in str(gm.graph).split('\n'):
        print(f"    {line}")

    print(f"\n  Detailed nodes:")
    for node in gm.graph.nodes:
        print(f"\n    [{node.op}] {node.name}")
        print(f"      target: {node.target}")

        # Output info
        val = node.meta.get("val")
        if val is not None:
            if isinstance(val, torch.Tensor):
                print(f"      output: shape={list(val.shape)}, dtype={val.dtype}")
            elif isinstance(val, (tuple, list)):
                for i, v in enumerate(val):
                    if isinstance(v, torch.Tensor):
                        print(f"      output[{i}]: shape={list(v.shape)}, dtype={v.dtype}")
            else:
                print(f"      output: {val}")

        # TensorSpec info (from SpecPropPass)
        spec = node.meta.get("spec")
        if spec is not None:
            if hasattr(spec, 'shape'):
                print(f"      spec: shape={spec.shape}, dtype={spec.dtype}")
                if hasattr(spec, 'mem_id'):
                    print(f"             mem_id={spec.mem_id}")
            elif isinstance(spec, (list, tuple)):
                for i, s in enumerate(spec):
                    if hasattr(s, 'shape'):
                        print(f"      spec[{i}]: shape={s.shape}, dtype={s.dtype}")

        # Vulkan memory metadata
        if "vkdg_memory_meta" in node.meta:
            mem_meta = node.meta["vkdg_memory_meta"]
            print(f"      vk_memory_meta: {mem_meta}")

        # Args
        for i, arg in enumerate(node.args):
            if hasattr(arg, 'name') and hasattr(arg, 'meta'):
                arg_val = arg.meta.get("val")
                if isinstance(arg_val, torch.Tensor):
                    is_const = arg.op in ("get_attr",) or "constant" in arg.name.lower()
                    const_str = " [CONSTANT]" if is_const else ""
                    print(f"      arg[{i}]: node={arg.name}, shape={list(arg_val.shape)}, dtype={arg_val.dtype}{const_str}")
                else:
                    print(f"      arg[{i}]: node={arg.name}, val={arg_val}")
            else:
                print(f"      arg[{i}]: {repr(arg)}")
        if node.kwargs:
            print(f"      kwargs: {node.kwargs}")

    if show_constants:
        # Show state dict / constants
        sd = gm.state_dict()
        if sd:
            print(f"\n  State dict ({len(sd)} entries):")
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}")
                    if v.numel() <= 64:
                        print(f"      value = {v.detach().cpu()}")

        # Named parameters/buffers
        for name, p in gm.named_parameters():
            print(f"  Param: {name}, shape={list(p.shape)}, dtype={p.dtype}")
            if p.numel() <= 64:
                print(f"    value = {p.detach().cpu()}")
        for name, b in gm.named_buffers():
            print(f"  Buffer: {name}, shape={list(b.shape)}, dtype={b.dtype}")
            if b.numel() <= 64:
                print(f"    value = {b.detach().cpu()}")


def print_vk_graph(vk_graph):
    """Print the VkGraph (Vulkan serialized graph) in detail."""
    from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
        VkTensor, VkDataType, VkStorageType, VkMemoryLayout,
        Int, Double, Bool, Null, IntList, DoubleList, BoolList, ValueList, String,
    )

    print(f"\n  VkGraph version: {vk_graph.version}")
    print(f"  Input IDs:  {vk_graph.input_ids}")
    print(f"  Output IDs: {vk_graph.output_ids}")
    print(f"  Storage type override: {vk_graph.storage_type_override}")
    print(f"  Memory layout override: {vk_graph.memory_layout_override}")
    print(f"  Num values: {len(vk_graph.values)}")
    print(f"  Num chain ops: {len(vk_graph.chain)}")
    print(f"  Num constants: {len(vk_graph.constants)}")

    print(f"\n  --- Values ---")
    for i, vk_val in enumerate(vk_graph.values):
        v = vk_val.value
        if isinstance(v, VkTensor):
            is_const = v.constant_id >= 0
            const_str = f", constant_id={v.constant_id}" if is_const else ""
            print(f"    Value[{i}]: VkTensor dims={v.dims}, dtype={VkDataType(v.datatype).name}, "
                  f"storage={VkStorageType(v.storage_type).name}, "
                  f"layout={VkMemoryLayout(v.memory_layout).name}, "
                  f"mem_obj={v.mem_obj_id}{const_str}")
        elif isinstance(v, Int):
            print(f"    Value[{i}]: Int = {v.int_val}")
        elif isinstance(v, Double):
            print(f"    Value[{i}]: Double = {v.double_val}")
        elif isinstance(v, Bool):
            print(f"    Value[{i}]: Bool = {v.bool_val}")
        elif isinstance(v, Null):
            print(f"    Value[{i}]: Null")
        elif isinstance(v, IntList):
            print(f"    Value[{i}]: IntList = {v.items}")
        elif isinstance(v, DoubleList):
            print(f"    Value[{i}]: DoubleList = {v.items}")
        elif isinstance(v, BoolList):
            print(f"    Value[{i}]: BoolList = {v.items}")
        elif isinstance(v, ValueList):
            print(f"    Value[{i}]: ValueList = {v.items}")
        elif isinstance(v, String):
            print(f"    Value[{i}]: String = {v.string_val}")
        else:
            print(f"    Value[{i}]: {type(v).__name__} = {v}")

    print(f"\n  --- Operator Chain ---")
    for i, op in enumerate(vk_graph.chain):
        arg_details = []
        for arg_id in op.args:
            vk_val = vk_graph.values[arg_id]
            v = vk_val.value
            if isinstance(v, VkTensor):
                is_const = v.constant_id >= 0
                const_tag = " [CONST]" if is_const else ""
                arg_details.append(f"v{arg_id}:Tensor{list(v.dims)}{const_tag}")
            elif isinstance(v, Int):
                arg_details.append(f"v{arg_id}:Int({v.int_val})")
            elif isinstance(v, Double):
                arg_details.append(f"v{arg_id}:Double({v.double_val})")
            elif isinstance(v, Bool):
                arg_details.append(f"v{arg_id}:Bool({v.bool_val})")
            elif isinstance(v, Null):
                arg_details.append(f"v{arg_id}:Null")
            elif isinstance(v, IntList):
                arg_details.append(f"v{arg_id}:IntList{v.items}")
            elif isinstance(v, DoubleList):
                arg_details.append(f"v{arg_id}:DoubleList{v.items}")
            else:
                arg_details.append(f"v{arg_id}:{type(v).__name__}")

        print(f"\n    Chain[{i}]: {op.name}  (node_id={op.node_id})")
        print(f"      args (value IDs): {op.args}")
        for j, detail in enumerate(arg_details):
            print(f"        arg[{j}]: {detail}")

    if vk_graph.constants:
        print(f"\n  --- Constant blobs ---")
        for i, c in enumerate(vk_graph.constants):
            print(f"    Constant[{i}]: offset={c.offset}, length={c.length}, key='{c.named_key}'")


def main():
    model = HardswishModel()
    model.eval()

    example_input = (torch.randn(1, 4, 8, 8),)

    # =========================================================================
    print("=" * 80)
    print("STEP 1: torch.export (ATen graph)")
    print("=" * 80)
    # =========================================================================

    exported = torch.export.export(model, example_input)
    print_graph_nodes(exported.graph_module, "ATen Graph")

    # Show lifted constants from exported program
    ep = exported
    if hasattr(ep, 'state_dict') and ep.state_dict:
        print(f"\n  EP state_dict:")
        for k, v in ep.state_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}, value={v.detach().cpu()}")

    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: to_edge (Edge IR graph)")
    print("=" * 80)
    # =========================================================================

    from executorch.exir import to_edge, EdgeCompileConfig

    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    edge_ep = edge.exported_program()
    print_graph_nodes(edge_ep.graph_module, "Edge IR Graph")

    if hasattr(edge_ep, 'state_dict') and edge_ep.state_dict:
        print(f"\n  EP state_dict:")
        for k, v in edge_ep.state_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}, value={v.detach().cpu()}")

    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Vulkan Backend Preprocess (manual, step by step)")
    print("=" * 80)
    # =========================================================================

    # Reproduce VulkanBackend.preprocess() manually to inspect each stage
    from executorch.backends.vulkan.vulkan_preprocess import (
        apply_passes, insert_prepack_nodes, unsafe_remove_auto_functionalized_pass,
        VkGraphBuilder, VkMemoryLayout, VkStorageType,
        serialize_vulkan_graph,
        FuseBatchNormPass, FusePatternsPass, FuseClampPass,
        AddmmToLinearTransform, RemoveRedundantOpsTransform,
        FuseQuantizedOpsTransform, FoldQDQPass, SqueezeUnsqueezeInputs,
        FuseViewCopyTransform, ViewCopyToSqueezeUnsqueezePass,
        RemoveAssertsTransform, TagMemoryMetaPass,
        SpecPropPass, ConstraintBasedSymShapeEvalPass,
        MemoryPlanningPass, MemoryPlanningAlgorithmSuite,
        greedy,
        DelegateMappingBuilder,
    )
    from executorch.exir.backend.backend_details import CompileSpec

    # Get the lowered module's original program
    vulkan_gm = edge.to_backend(
        __import__('executorch.backends.vulkan.partitioner.vulkan_partitioner',
                   fromlist=['VulkanPartitioner']).VulkanPartitioner(
            compile_options={"texture_limits": (2048, 2048, 2048)},
        )
    ).exported_program().graph_module

    # Find the lowered module
    lowered_mod = None
    for name, mod in vulkan_gm.named_modules():
        if type(mod).__name__ == "LoweredBackendModule":
            lowered_mod = mod
            break

    if lowered_mod is not None:
        print(f"\n  Found LoweredBackendModule")
        print(f"  Backend ID: {lowered_mod.backend_id}")

        # The original_module has the graph BEFORE vulkan preprocess
        orig = lowered_mod.original_module
        print_graph_nodes(orig.graph_module, "Original module BEFORE Vulkan preprocess")

        # Now manually run the Vulkan preprocess pipeline on a copy
        program = copy.deepcopy(orig)

        texture_limits = (2048, 2048, 2048)

        print(f"\n  --- Applying fusion passes ---")
        program = apply_passes(
            program,
            [
                FuseBatchNormPass(program),
                FusePatternsPass(),
                FuseClampPass(),
                AddmmToLinearTransform(),
                RemoveRedundantOpsTransform(),
                FuseQuantizedOpsTransform(),
                FoldQDQPass(),
                SqueezeUnsqueezeInputs(),
                FuseViewCopyTransform(),
                ViewCopyToSqueezeUnsqueezePass(),
            ],
        )
        print_graph_nodes(program.graph_module, "After fusion passes")

        print(f"\n  --- Applying SpecPropPass ---")
        program = apply_passes(program, [SpecPropPass()])
        print_graph_nodes(program.graph_module, "After SpecPropPass")

        print(f"\n  --- Applying insert_prepack_nodes + RemoveAssertsTransform ---")
        program = apply_passes(
            program,
            [
                RemoveAssertsTransform(),
                insert_prepack_nodes,
            ],
        )
        print_graph_nodes(program.graph_module, "After insert_prepack_nodes (KEY STEP)")

        print(f"\n  --- Applying TagMemoryMetaPass ---")
        program = apply_passes(
            program,
            [
                TagMemoryMetaPass(
                    texture_limits,
                    default_storage_type=VkStorageType.TEXTURE_3D,
                    default_memory_layout=VkMemoryLayout.TENSOR_WIDTH_PACKED,
                    force_fp16=False,
                ),
            ],
        )
        print_graph_nodes(program.graph_module, "After TagMemoryMetaPass")

        print(f"\n  --- Applying memory planning ---")
        greedy_memory_planning = partial(
            greedy, allow_overlapping_allocations=False
        )
        mem_planning_suite = MemoryPlanningAlgorithmSuite(
            algo_list=[greedy_memory_planning]
        )
        program.graph_module.encounter_to_out_var_failure = True
        program = apply_passes(
            program,
            [
                ConstraintBasedSymShapeEvalPass(),
                MemoryPlanningPass(memory_planning_algo=mem_planning_suite),
            ],
        )
        print_graph_nodes(program.graph_module, "After memory planning (FINAL graph)")

        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 4: Build VkGraph")
        print("=" * 80)
        # =====================================================================

        graph_builder = VkGraphBuilder(
            program,
            DelegateMappingBuilder(generated_identifiers=True),
            downcast_64_bit=True,
            force_fp16=False,
        )
        vk_graph = graph_builder.build_graph()

        print_vk_graph(vk_graph)

        # Also print constant tensor values from the graph builder
        print(f"\n  --- Constant tensors from graph_builder ---")
        if hasattr(graph_builder, 'const_tensors'):
            for i, ct in enumerate(graph_builder.const_tensors):
                if isinstance(ct, torch.Tensor):
                    print(f"    const_tensor[{i}]: shape={list(ct.shape)}, dtype={ct.dtype}")
                    if ct.numel() <= 64:
                        print(f"      value = {ct.detach().cpu().numpy()}")

    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Serialize and save")
    print("=" * 80)
    # =========================================================================

    # Use the already-partitioned edge program
    et_program = edge.to_backend(
        __import__('executorch.backends.vulkan.partitioner.vulkan_partitioner',
                   fromlist=['VulkanPartitioner']).VulkanPartitioner(
            compile_options={"texture_limits": (2048, 2048, 2048)},
        )
    ).to_executorch()

    import os
    out_path = os.path.join(os.path.dirname(__file__), "hardswish_inspect.pte")
    with open(out_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"  Serialized: {len(et_program.buffer)} bytes -> {out_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
