#!/usr/bin/env python3
"""
=============================================================================
 RiverOne-QC-4B-v2 AQLM 量化 — 2×16 scheme, 后8层, MLP only
 ★ Codebook 跨层共享 ★
 
 流程:
   1. 标准逐层量化 (与 baseline 相同)
   2. 后处理: codebook 按投影类型共享 (平均 + NN 重分配)
   
 预期节省: ~84MB (24套独立 codebook → 3套共享)
=============================================================================
"""

import os
import sys
import json
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---- Step 1: Standard quantization ----
print("=" * 60)
print("Phase 1: Standard AQLM Quantization (8L MLP-only)")
print("=" * 60)

std_script = os.path.join(SCRIPT_DIR, "quantize_2x16_8L_mlp_only.py")

# Import and run standard quantization
sys.path.insert(0, SCRIPT_DIR)

# Ensure restore of clean state: if the baseline output already exists, 
# we'll use the post-processing directly without re-quantizing
import importlib.util
spec = importlib.util.spec_from_file_location("_baseline_quant", std_script)
baseline = importlib.util.module_from_spec(spec)

# Override OUTPUT_DIR temporarily — the standard quant goes to its usual location
# Post-processing will read from there and write to shared location

# Check if baseline model already exists
baseline_output = os.path.join(PROJECT_ROOT, "RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly")
if os.path.exists(os.path.join(baseline_output, "model.safetensors.index.json")):
    print(f"Baseline model already exists at {baseline_output}, skipping quantization.")
    print("(Delete it if you want to re-quantize)")
else:
    print("Running standard quantization...")
    spec.loader.exec_module(baseline)

# ---- Step 2: Codebook Sharing Post-Processing ----
print("\n" + "=" * 60)
print("Phase 2: Codebook Sharing Post-Processing")
print("=" * 60)

shared_output = os.path.join(PROJECT_ROOT, "RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly-shared")

# Run post-processing inline
def run_sharing():
    """Run codebook sharing inline."""
    import torch
    import safetensors.torch as st
    from collections import defaultdict
    from pathlib import Path
    import shutil

    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "engine"))
    from src.kmeans import find_nearest_cluster

    model_dir = Path(baseline_output)
    output_dir = Path(shared_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    with open(model_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Read quant config
    with open(model_dir / "quant_config.json") as f:
        qc = json.load(f)

    # Group AQLM keys by projection type
    projection_groups: dict[str, list[str]] = defaultdict(list)
    for key in weight_map:
        if key.endswith(".codebooks"):
            parts = key.split(".")
            # language_model.model.layers.N.mlp.PROJ.codebooks
            for i, p in enumerate(parts):
                if p == "mlp" and i + 1 < len(parts):
                    proj = parts[i + 1]
                    projection_groups[proj].append(key)
                    break

    print(f"Found projection groups: {list(projection_groups.keys())}")
    for proj, keys in projection_groups.items():
        print(f"  {proj}: {len(keys)} layers")

    # Load all shards containing AQLM keys
    all_shards = set()
    for key in weight_map:
        if any(key.endswith(s) for s in (".codebooks", ".codes", ".scales")):
            all_shards.add(weight_map[key])
    # Also all other shards for copying
    all_shards |= set(weight_map.values())

    shard_data = {}
    for shard_name in sorted(all_shards):
        shard_path = model_dir / shard_name
        if shard_path.exists():
            shard_data[shard_name] = {
                k: v.clone() for k, v in st.load_file(str(shard_path)).items()
            }

    # For each projection type: average codebooks, reassign codes
    total_saved = 0
    for proj_type, cb_keys in projection_groups.items():
        print(f"\n--- Sharing {proj_type} ---")
        # Collect codebooks from all layers
        codebooks_list = []
        layer_keys = []
        for cb_key in cb_keys:
            shard = shard_data[weight_map[cb_key]]
            codebooks_list.append(shard[cb_key])
            # Find corresponding codes and scales keys
            base = cb_key[:-len(".codebooks")]
            codes_key = base + ".codes"
            scales_key = base + ".scales"
            layer_keys.append((cb_key, codes_key, scales_key))

        stacked_cb = torch.stack(codebooks_list)
        shared_cb = stacked_cb.mean(dim=0)
        cb_size_before = sum(cb.numel() * cb.element_size() for cb in codebooks_list)
        cb_size_after = shared_cb.numel() * shared_cb.element_size() * len(codebooks_list)
        total_saved += cb_size_before - cb_size_after

        # For each layer: reconstruct weight, find nearest shared codebook entries
        for cb_key, codes_key, scales_key in layer_keys:
            shard_name = weight_map[cb_key]
            shard = shard_data[shard_name]

            cb = shared_cb.float()
            codes = shard[codes_key]  # int16
            scales = shard[scales_key].float()

            nc, cs, o_gs, i_gs = cb.shape
            og, ig, _ = codes.shape

            # Reconstruct current weight patches
            with torch.no_grad():
                weight_patches_list = []
                for ci in range(nc):
                    cb_entries = cb[ci][codes[:, :, ci].long()]  # [og, ig, o_gs, i_gs]
                    weight_patches_list.append(cb_entries)
                weight_patches = sum(weight_patches_list)
                # scales: [og, 1, 1, 1] → broadcast multiply
                weight_patches = weight_patches * scales

                data = weight_patches.reshape(og * ig, o_gs * i_gs)
                shared_cb_flat = shared_cb.reshape(nc, cs, o_gs * i_gs).float()

                # Residual NN search
                residual = data.clone()
                new_codes_list = []
                for ci in range(nc):
                    nearest, recon = find_nearest_cluster(residual.float(), shared_cb_flat[ci].float())
                    new_codes_list.append(nearest.reshape(og, ig, 1))
                    residual = residual - recon

                new_codes = torch.cat(new_codes_list, dim=-1)

            # Replace in shard
            shard[cb_key] = shared_cb.to(shard[cb_key].dtype)
            shard[codes_key] = new_codes.to(shard[codes_key].dtype)
            # scales kept as-is

            old_codes = codes.long()
            same = (old_codes == new_codes).float().mean().item()
            layer_id = cb_key.split(".layers.")[1].split(".")[0]
            print(f"  L{layer_id} {proj_type}: codes match={same:.1%}")

    # Save all shards
    print(f"\nSaving to {output_dir}...")
    for shard_name, tensors in shard_data.items():
        out_path = output_dir / shard_name
        st.save_file(tensors, str(out_path))

    # Copy non-shard files
    for item in model_dir.iterdir():
        if item.name.startswith("model-") and item.name.endswith(".safetensors"):
            continue  # Already saved
        if item.is_file() and item.name != "model.safetensors.index.json":
            shutil.copy2(str(item), str(output_dir / item.name))

    # Write index
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Update quant_config
    qc["scheme"] = "2x16-MLPonly-8L-shared"
    qc["codebook_sharing"] = {
        "method": "average + NN reassign",
        "shared_by": "projection_type",
        "codebooks_before": sum(len(v) for v in projection_groups.values()),
        "codebooks_after": len(projection_groups),
        "savings_mb": total_saved / 1e6,
    }
    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(qc, f, indent=2)

    print(f"\nDone! Shared model saved to {output_dir}")
    print(f"Codebook storage saved: {total_saved/1e6:.1f} MB")


run_sharing()
print("\nAll done!")
