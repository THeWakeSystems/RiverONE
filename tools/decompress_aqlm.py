#!/usr/bin/env python3
"""
Decompress AQLM model to standard checkpoint for full SFT training.

AQLM layers store codes/codebooks/scales instead of standard .weight.
This script decompresses them and saves a standard safetensors checkpoint.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

MODEL_DIR = "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-MLPonly"
OUTPUT_DIR = "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-decompressed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

from aqlm.utils import _dequantize_weight, unpack_int_data

# Load index to find which shard each key is in
with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)

weight_map = index["weight_map"]

# Group keys by shard file
shard_files = set(weight_map.values())
print(f"Found {len(shard_files)} shard files: {sorted(shard_files)}")

# Track AQLM keys for each layer
aqlm_keys = {}  # layer_idx -> {projection: {codes, codebooks, scales}}
for key in sorted(weight_map.keys()):
    if ".codes" in key or ".codebooks" in key or ".scales" in key:
        # Extract layer and projection info
        parts = key.split(".")
        layer_idx = None
        proj = None
        # Find layer index
        for i, p in enumerate(parts):
            if p == "layers":
                layer_idx = int(parts[i + 1])
                break
        # Find projection type
        for p in ["down_proj", "gate_proj", "up_proj"]:
            if p in key:
                proj = p
                break
        if layer_idx is not None and proj is not None:
            if layer_idx not in aqlm_keys:
                aqlm_keys[layer_idx] = {}
            if proj not in aqlm_keys[layer_idx]:
                aqlm_keys[layer_idx][proj] = {}
            
            # Determine type
            if ".codes" in key:
                aqlm_keys[layer_idx][proj]["codes"] = key
            elif ".codebooks" in key:
                aqlm_keys[layer_idx][proj]["codebooks"] = key
            elif ".scales" in key:
                aqlm_keys[layer_idx][proj]["scales"] = key

print(f"\nFound AQLM layers: {sorted(aqlm_keys.keys())}")

# Load all shards
all_tensors = {}
for shard_file in sorted(shard_files):
    path = os.path.join(MODEL_DIR, shard_file)
    print(f"Loading {shard_file}...")
    tensors = load_file(path)
    all_tensors.update(tensors)

print(f"\nTotal tensors loaded: {len(all_tensors)}")

# Decompress AQLM layers
decompressed = {}
for layer_idx in sorted(aqlm_keys.keys()):
    for proj in ["down_proj", "gate_proj", "up_proj"]:
        aqlm_info = aqlm_keys[layer_idx][proj]
        codes_key = aqlm_info["codes"]
        codebooks_key = aqlm_info["codebooks"]
        scales_key = aqlm_info["scales"]
        
        codes = all_tensors[codes_key]
        codebooks = all_tensors[codebooks_key]
        scales = all_tensors[scales_key]
        
        print(f"Decompressing layer {layer_idx} {proj}: codes={codes.shape}, codebooks={codebooks.shape}, scales={scales.shape}")
        
        # Convert signed int16 codes to unsigned indices
        # AQLM stores codes as int16 (-32768..32767), but _dequantize_weight expects 0..65535
        if codes.dtype == torch.int16 and codes.min() < 0:
            codes = codes.to(torch.int64) + 32768
            print(f"  shifted codes to unsigned: dtype={codes.dtype}, range=[{codes.min().item()},{codes.max().item()}]")
        elif codes.dtype == torch.int32:
            codes = unpack_int_data(codes, 16)
            codes = codes.to(torch.int64)
            print(f"  unpacked codes: {codes.shape}, dtype={codes.dtype}")
        
        # Decompress
        weight = _dequantize_weight(codes, codebooks, scales)
        print(f"  -> weight shape: {weight.shape}")
        
        # Create standard weight key
        weight_key = codes_key.replace(".codes", ".weight")
        decompressed[weight_key] = weight
        
        # Remove old keys
        del all_tensors[codes_key]
        del all_tensors[codebooks_key]
        del all_tensors[scales_key]

# Add decompressed weights to all_tensors
all_tensors.update(decompressed)

# Save output
# Re-create shard files matching original structure
# For simplicity, save everything into one big file
print(f"\nTotal tensors after decompression: {len(all_tensors)}")
print("Saving decompressed model...")

# Split into shards similar to original
SHARD_SIZE = 3_000_000_000  # ~3B per shard (in parameter elements)

# Calculate total elements
total_elements = 0
for key, tensor in all_tensors.items():
    total_elements += tensor.numel()

print(f"Total elements: {total_elements:,}")

# Group into shards
shard_index = 0
current_shard = {}
current_elements = 0
new_weight_map = {}

for key in sorted(all_tensors.keys()):
    tensor = all_tensors[key]
    nelem = tensor.numel()
    
    if current_elements > 0 and current_elements + nelem > SHARD_SIZE:
        # Save current shard
        shard_name = f"model-{shard_index + 1:05d}-of-NNN.safetensors"
        save_file(current_shard, os.path.join(OUTPUT_DIR, shard_name))
        print(f"  Saved {shard_name}: {len(current_shard)} tensors, {current_elements:,} elements")
        shard_index += 1
        current_shard = {}
        current_elements = 0
    
    shard_name_placeholder = f"model-{shard_index + 1:05d}-of-NNN.safetensors"
    current_shard[key] = tensor
    new_weight_map[key] = shard_name_placeholder
    current_elements += nelem

# Save last shard
if current_shard:
    shard_name = f"model-{shard_index + 1:05d}-of-NNN.safetensors"
    save_file(current_shard, os.path.join(OUTPUT_DIR, shard_name))
    print(f"  Saved {shard_name}: {len(current_shard)} tensors, {current_elements:,} elements")
    shard_index += 1

# Update placeholder names with actual count
total_shards = shard_index
for key in new_weight_map:
    new_weight_map[key] = new_weight_map[key].replace("-of-NNN", f"-of-{total_shards:05d}")

# Rename files
for i in range(total_shards):
    old_name = f"model-{i + 1:05d}-of-NNN.safetensors"
    new_name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
    os.rename(os.path.join(OUTPUT_DIR, old_name), os.path.join(OUTPUT_DIR, new_name))

# Save index
index_content = {
    "metadata": {"total_size": total_elements * 2},  # approx bytes for bf16
    "weight_map": new_weight_map
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index_content, f, indent=2)

# Copy config files
import shutil
for fname in os.listdir(MODEL_DIR):
    if fname.endswith(".json") or fname.endswith(".jinja") or fname.endswith(".txt") or fname.endswith(".py"):
        if fname != "model.safetensors.index.json" and fname != "quant_config.json":
            src = os.path.join(MODEL_DIR, fname)
            dst = os.path.join(OUTPUT_DIR, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

print(f"\nDone! Decompressed model saved to {OUTPUT_DIR}")
print(f"Total shards: {total_shards}")
