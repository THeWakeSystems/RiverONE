#!/usr/bin/env python3
"""
Dequantize AQLM model → regular safetensors weights.
Reads codebooks/codes/scales from the AQLM model, dequantizes to Linear weights,
and writes back as a regular model that can be loaded normally.
"""
from __future__ import annotations

import json, sys, os, shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from aqlm.utils import _dequantize_weight, unpack_int_data

MODEL_PATH = Path("/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-36L")
OUTPUT_PATH = Path("/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-36L-dequantized")

# AQLM config
IN_GROUP_SIZE = 16
OUT_GROUP_SIZE = 1
NUM_CODEBOOKS = 1
NBITS = 16

def load_all_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from all safetensors shards."""
    tensors = {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = set(weight_map.values())
        for shard in sorted(shards):
            shard_path = model_dir / shard
            print(f"  Loading {shard} ...")
            with safe_open(str(shard_path), framework="pt") as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
    else:
        sf_path = model_dir / "model.safetensors"
        with safe_open(str(sf_path), framework="pt") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)
    return tensors

def group_aqlm_tensors(tensors: dict) -> dict[str, dict[str, torch.Tensor]]:
    """Group AQLM params: module_path → {codebooks, codes, scales}."""
    groups = defaultdict(dict)
    for key in tensors:
        for suffix in (".codebooks", ".codes", ".scales"):
            if key.endswith(suffix):
                module_path = key[: -len(suffix)]
                param_type = suffix.lstrip(".")
                groups[module_path][param_type] = tensors[key]
                break
    return dict(groups)

def main():
    print("=== Step 1: Loading all tensors ===")
    tensors = load_all_tensors(MODEL_PATH)
    print(f"Total tensors: {len(tensors)}")
    
    print("\n=== Step 2: Grouping AQLM params ===")
    aqlm_groups = group_aqlm_tensors(tensors)
    print(f"AQLM modules: {len(aqlm_groups)}")
    
    print("\n=== Step 3: Dequantizing ===")
    new_tensors = {}
    dequantized = 0
    
    for key, tensor in tensors.items():
        # Skip AQLM-specific keys (codebooks, codes, scales) — they're replaced
        if any(key.endswith(s) for s in (".codebooks", ".codes", ".scales")):
            continue
        new_tensors[key] = tensor
    
    for module_path, aqlm_params in aqlm_groups.items():
        cb = aqlm_params["codebooks"].float()
        codes_raw = aqlm_params["codes"]
        codes = unpack_int_data(codes_raw, NBITS)  # int16 → int64
        scales = aqlm_params["scales"].float()
        
        # Dequantize
        weight = _dequantize_weight(codes, cb, scales)
        
        # Store as regular .weight
        weight_key = f"{module_path}.weight"
        new_tensors[weight_key] = weight.to(torch.bfloat16)
        dequantized += 1
        
        if dequantized <= 3:
            print(f"  {weight_key}: {list(weight.shape)} min={weight.min():.4f} max={weight.max():.4f}")
    
    print(f"Dequantized {dequantized} layers")
    print(f"New tensor count: {len(new_tensors)}")
    
    print(f"\n=== Step 4: Writing to {OUTPUT_PATH} ===")
    # Copy all non-weight files
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for item in MODEL_PATH.iterdir():
        if item.name.endswith(".safetensors") or item.name == "model.safetensors.index.json":
            continue
        dst = OUTPUT_PATH / item.name
        if item.is_dir():
            if not dst.exists():
                shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    
    # Write new safetensors
    # Split into shards of ~4GB each
    MAX_SHARD_SIZE = 4 * 1024 * 1024 * 1024  # 4GB
    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 1
    
    for key in sorted(new_tensors.keys()):
        t = new_tensors[key]
        t_size = t.numel() * t.element_size()
        if current_size + t_size > MAX_SHARD_SIZE and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        current_shard[key] = t
        current_size += t_size
    
    if current_shard:
        shards.append(current_shard)
    
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        shard_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        shard_path = OUTPUT_PATH / shard_name
        save_file(shard, str(shard_path))
        for key in shard:
            weight_map[key] = shard_name
        size_mb = sum(t.numel() * t.element_size() for t in shard.values()) / 1024 / 1024
        print(f"  {shard_name}: {len(shard)} tensors, {size_mb:.0f} MB")
    
    # Write index
    index_path = OUTPUT_PATH / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in new_tensors.values())},
                   "weight_map": weight_map}, f, indent=2)
    
    total_mb = sum(t.numel() * t.element_size() for t in new_tensors.values()) / 1024 / 1024
    print(f"\nTotal: {total_mb:.0f} MB, {len(shards)} shards")
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Quick verification
    print("\n=== Step 5: Quick load test ===")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    
    import torch as t
    from transformers import AutoModel
    
    device = t.device("cuda:0")
    model = AutoModel.from_pretrained(
        str(OUTPUT_PATH), torch_dtype=t.bfloat16, trust_remote_code=True, device_map=None
    ).to(device).eval()
    
    lm = model.language_model
    layer0 = lm.model.layers[0]
    q_proj = layer0.self_attn.q_proj
    print(f"q_proj type: {type(q_proj).__name__}")
    if hasattr(q_proj, 'weight'):
        print(f"  weight shape: {q_proj.weight.shape}, values: {q_proj.weight[0,:5]}")
    
    # Quick inference
    data_root = Path("/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/Data")
    with open(data_root / "q2.jsonl") as f:
        sample = json.loads(f.readline())
    
    import sys
    sys.path.insert(0, str(Path("/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/scripts")))
    from riverone_backend import load_image_tensor
    from transformers import AutoTokenizer
    
    img_path = data_root / sample["image_path"]
    pv, nt = load_image_tensor(img_path, max_num=6, device=device, dtype=t.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_PATH), trust_remote_code=True, use_fast=False)
    
    with t.inference_mode():
        response = model.chat(
            tokenizer=tokenizer, pixel_values=pv, question=sample["prompt"],
            generation_config={"max_new_tokens": 128, "do_sample": False},
            num_patches_list=[nt],
        )
    
    print(f"ID: {sample['id']}")
    print(f"GT (first 150):  {sample['answer'][:150]}")
    print(f"Pred (first 300): {str(response)[:300]}")
    print("\n✓ Done!")

if __name__ == "__main__":
    main()
