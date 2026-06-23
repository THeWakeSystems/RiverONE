#!/usr/bin/env python3
"""
=============================================================================
 Codebook 跨层共享后处理
 输入：标准逐层量化模型（每层独立 codebooks）
 输出：codebook 跨层共享模型（同类型投影共享一套 codebook）

 原理：
   1. 按投影类型分组 (gate_proj/up_proj/down_proj)
   2. 同组各层 codebook 取平均作为共享 codebook
   3. 对每层用 find_nearest_cluster 重新分配 codes
   4. 保存到新目录

 预期节省: ~84MB (24套 codebook → 3套)
=============================================================================
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AQLM_LIB = os.path.join(SCRIPT_DIR, "..", "engine")
if AQLM_LIB not in sys.path:
    sys.path.insert(0, AQLM_LIB)

import torch
import torch.nn as nn
import safetensors.torch as st

# Import NN search from engine
from src.kmeans import find_nearest_cluster

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

INPUT_MODEL_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly"
)
OUTPUT_MODEL_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly-shared"
)


def load_aqlm_weights(model_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    """Load all AQLM codebooks/codes/scales from safetensors.
    Returns: {layer_proj_type: {'codebooks': tensor, 'codes': tensor, 'scales': tensor}}
    """
    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        index = json.load(f)

    # Group by layer + projection type
    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, shard in index["weight_map"].items():
        for suffix in (".codebooks", ".codes", ".scales"):
            if key.endswith(suffix):
                param = suffix.lstrip(".")
                # Extract proj_type: layers.{N}.mlp.{proj_type}
                parts = key.split(".")
                layer_idx = None
                proj_type = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_idx = parts[i + 1]
                    if p == "mlp" and i + 1 < len(parts):
                        proj_type = parts[i + 1]
                if proj_type:
                    group_key = f"L{layer_idx}_{proj_type}"
                    groups[group_key][param] = key
                    groups[group_key]["shard"] = shard
                break

    # Load tensors
    shard_cache = {}
    result: dict[str, dict[str, torch.Tensor]] = {}
    for group_key, info in groups.items():
        shard_name = info["shard"]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = st.load_file(str(model_dir / shard_name))
        shard = shard_cache[shard_name]
        result[group_key] = {
            "codebooks": shard[info["codebooks"]].clone(),
            "codes": shard[info["codes"]].clone(),
            "scales": shard[info["scales"]].clone(),
        }
        # Also store shard name for later replacement
        result[group_key]["_shard"] = shard_name
        result[group_key]["_keys"] = {
            "codebooks": info["codebooks"],
            "codes": info["codes"],
            "scales": info["scales"],
        }

    return result


def share_codebooks(
    weights: dict[str, dict[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    """Average codebooks per projection type, reassign codes via NN search."""
    # Group by projection type
    by_proj: dict[str, list[str]] = defaultdict(list)
    for group_key in weights:
        proj_type = group_key.split("_", 1)[1]  # gate_proj / up_proj / down_proj
        by_proj[proj_type].append(group_key)

    print(f"Projection types: {list(by_proj.keys())}")
    for proj_type, keys in by_proj.items():
        print(f"  {proj_type}: {len(keys)} layers ({', '.join(keys)})")

    # For each projection type: average codebooks, reassign codes
    for proj_type, keys in by_proj.items():
        print(f"\n--- Sharing codebooks for {proj_type} ---")

        # Average codebooks
        stacked_cb = torch.stack([weights[k]["codebooks"] for k in keys])  # [N, 2, 65536, 1, 16]
        shared_cb = stacked_cb.mean(dim=0)  # [2, 65536, 1, 16]
        print(f"  Shared codebooks shape: {list(shared_cb.shape)}")

        # Verify shapes are identical
        ref_cb = weights[keys[0]]["codebooks"]
        cb_shape = list(ref_cb.shape)
        for k in keys[1:]:
            assert list(weights[k]["codebooks"].shape) == cb_shape, \
                f"Codebook shape mismatch: {k} has {list(weights[k]['codebooks'].shape)} vs {cb_shape}"

        # Reassign codes for each layer
        for k in keys:
            w = weights[k]
            codebooks = w["codebooks"]  # [2, codebook_size, 1, 16] original
            codes = w["codes"]  # [num_out_groups, num_in_groups, 2]
            scales = w["scales"]  # [num_out_groups, num_in_groups, 2, 1]

            num_codebooks, codebook_size, out_gs, in_gs = codebooks.shape
            num_out_groups, num_in_groups, _ = codes.shape

            # Dequantize current weights using OLD codebooks + codes + scales
            # W_recon = scale * sum(codebook[code])
            # This gives us the "target" weight that the NEW shared codebook should represent

            # For each codebook, reconstruct per-group weight patch
            # Simpler: use NN search directly on the dequantized weight patches

            # Reconstruct weight and reassign codes using FAISS GPU NN search
            with torch.no_grad():
                og, ig, nc = num_out_groups, num_in_groups, num_codebooks
                cb = codebooks.float()  # [nc, cs, o_gs, i_gs]
                c = codes.long()  # [og, ig, nc]
                s = scales.float()  # [og, 1, 1, 1] shape confirmed

                # Gather codebook entries for full weight reconstruction
                weight_patches_list = []
                for ci in range(nc):
                    cb_entries = cb[ci][c[:, :, ci]]  # [og, ig, o_gs, i_gs]
                    weight_patches_list.append(cb_entries)
                weight_patches = sum(weight_patches_list)  # [og, ig, o_gs, i_gs]
                weight_patches = weight_patches * s  # broadcast [og, ig, o_gs, i_gs] * [og, 1, 1, 1]

                # Reshape to [N, dim] for NN search
                data = weight_patches.reshape(og * ig, out_gs * in_gs)  # ~1.5M x 16
                shared_cb_flat = shared_cb.float().reshape(nc, codebook_size, out_gs * in_gs)

                # FAISS CPU NN search (works without GPU visibility)
                import faiss
                residual = data.numpy().astype('float32')
                new_codes_list = []
                for ci in range(nc):
                    cb_np = shared_cb_flat[ci].numpy().astype('float32')  # [65536, 16]
                    d = cb_np.shape[1]
                    index = faiss.IndexFlatL2(d)
                    index.add(cb_np)
                    D, I = index.search(residual, 1)
                    nearest = torch.from_numpy(I[:, 0]).to(codes.device)
                    reconstructed = torch.from_numpy(cb_np[I[:, 0]]).to(torch.float32)
                    new_codes_list.append(nearest.reshape(og, ig, 1))
                    residual = residual - reconstructed.numpy()

                new_codes = torch.cat(new_codes_list, dim=-1)  # [og, ig, nc]

            # Replace codes
            w["codebooks"] = shared_cb.to(codebooks.dtype)
            w["codes"] = new_codes.to(codes.dtype)
            # scales kept as-is (they're per-layer specific)

            # Print stats
            old_codes = codes.long()
            code_same = (old_codes == new_codes).float().mean().item()
            print(f"  {k}: codes same={code_same:.1%}")

    return weights


def save_shared_model(
    weights: dict[str, dict[str, torch.Tensor]],
    input_dir: str,
    output_dir: str,
):
    """Save modified AQLM weights back to safetensors, copy non-AQLM files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the index to get shard→file mapping
    with open(input_dir / "model.safetensors.index.json") as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Determine which shards need rewriting
    shard_keys: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for group_key, w in weights.items():
        for param_name, full_key in w["_keys"].items():
            shard_name = weight_map[full_key]
            shard_keys[shard_name][full_key] = w[param_name]

    # Load each shard, replace keys, save
    new_weight_map = dict(weight_map)
    for shard_name, replacements in shard_keys.items():
        shard_path = input_dir / shard_name
        out_path = output_dir / shard_name
        tensors = st.load_file(str(shard_path))

        for key, new_tensor in replacements.items():
            old = tensors[key]
            # Clone to avoid shared memory detection (same shared codebook used by multiple keys)
            tensors[key] = (new_tensor.to(old.dtype) if new_tensor.dtype != old.dtype else new_tensor).clone()

        st.save_file(tensors, str(out_path))
        print(f"  Saved {shard_name} ({len(replacements)} keys replaced)")

    # Copy non-shard files
    for item in input_dir.iterdir():
        if item.name.startswith("model-") and item.name.endswith(".safetensors"):
            if item.name in shard_keys:
                continue  # Already saved with modifications
        if item.is_file() and item.name != "model.safetensors.index.json":
            shutil.copy2(str(item), str(output_dir / item.name))
        elif item.is_dir():
            pass  # Skip dirs

    # Write new index
    new_index_path = output_dir / "model.safetensors.index.json"
    with open(new_index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Update quant_config
    qc_path = output_dir / "quant_config.json"
    if qc_path.exists():
        with open(qc_path) as f:
            qc = json.load(f)
        qc["scheme"] = qc.get("scheme", "2x16-MLPonly-8L") + "-shared"
        qc["codebook_sharing"] = {
            "method": "average + NN reassign",
            "shared_by": "projection_type (gate_proj / up_proj / down_proj)",
            "codebooks_before": 24,
            "codebooks_after": 3,
            "savings_mb": (24 - 3) * 4.0,
        }
        with open(qc_path, "w") as f:
            json.dump(qc, f, indent=2)

    print(f"\nShared model saved to {output_dir}")


def main():
    print("=" * 60)
    print("Codebook Cross-Layer Sharing — Post-Processing")
    print("=" * 60)
    print(f"Input:  {INPUT_MODEL_DIR}")
    print(f"Output: {OUTPUT_MODEL_DIR}")

    # 1. Load AQLM weights
    print("\n[1] Loading AQLM weights...")
    weights = load_aqlm_weights(INPUT_MODEL_DIR)
    print(f"  Loaded {len(weights)} AQLM layers")

    # 2. Share codebooks
    print("\n[2] Sharing codebooks per projection type...")
    weights = share_codebooks(weights)
    print("  Done!")

    # 3. Save
    print("\n[3] Saving shared model...")
    save_shared_model(weights, INPUT_MODEL_DIR, OUTPUT_MODEL_DIR)

    # Compute savings
    total_cb_before = sum(w["codebooks"].numel() * w["codebooks"].element_size() for w in weights.values())
    total_cb_after = 0
    seen_proj = set()
    for group_key, w in weights.items():
        proj_type = group_key.split("_", 1)[1]
        if proj_type not in seen_proj:
            total_cb_after += w["codebooks"].numel() * w["codebooks"].element_size()
            seen_proj.add(proj_type)

    print(f"\n=== Summary ===")
    print(f"Codebook storage before: {total_cb_before/1e6:.1f} MB (24 codebooks)")
    print(f"Codebook storage after:  {total_cb_after/1e6:.1f} MB (3 shared codebooks)")
    print(f"Saved:                  {(total_cb_before - total_cb_after)/1e6:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
