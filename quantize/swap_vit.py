#!/usr/bin/env python3
"""ViT Swap: Replace miniViT_distilled ViT with AQLM model's full ViT.

Source models:
  - ViT donor: AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-Attn14MLP16 (full ViT, block 24 has own weights)
  - LLM donor: /home/lxy/miniViT_distilled/ (bf16 LLM, miniViT'd ViT with weight sharing)

Output: new model with full ViT from AQLM + LLM from miniViT_distilled.
Existing model directories are NOT modified.
"""
from __future__ import annotations
import json, shutil, os, re
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file

# --- Config ---
VIT_DONOR = Path("/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-Attn14MLP16")
LLM_DONOR = Path("/home/lxy/miniViT_distilled")
OUTPUT_DIR = Path("/home/lxy/miniViT_distilled_fullViT")

# --- 1. Create output directory ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[1/5] Output directory: {OUTPUT_DIR}")

# --- 2. Copy config/tokenizer files from LLM donor ---
print("[2/5] Copying config files from LLM donor...")
COPY_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
    "configuration_riverone_qc.py", "modeling_riverone_qc.py", "modeling_ising_vit.py",
    "conversation.py", "preprocessor_config.json", "processor_config.json",
    "chat_template.jinja", "video_preprocessor_config.json",
]
for fname in COPY_FILES:
    src = LLM_DONOR / fname
    if src.exists():
        shutil.copy2(str(src), str(OUTPUT_DIR / fname))

# Handle miniViT_config.json — use from VIT donor (reflects the full ViT structure)
mini_config_src = VIT_DONOR / "miniViT_config.json"
if mini_config_src.exists():
    shutil.copy2(str(mini_config_src), str(OUTPUT_DIR / "miniViT_config.json"))
    print(f"  Copied miniViT_config.json from ViT donor")
elif (LLM_DONOR / "miniViT_config.json").exists():
    # If ViT donor has no miniViT config, copy from LLM donor but update it
    with open(LLM_DONOR / "miniViT_config.json") as f:
        cfg = json.load(f)
    # Update to indicate NO weight sharing (full ViT)
    cfg["weight_sharing"] = False
    cfg.pop("shared_blocks", None)
    cfg.pop("transform_params", None)
    with open(OUTPUT_DIR / "miniViT_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Updated miniViT_config.json (full ViT, no weight sharing)")

# Also copy quant_config.json from ViT donor if it exists
quant_src = VIT_DONOR / "quant_config.json"
if quant_src.exists():
    shutil.copy2(str(quant_src), str(OUTPUT_DIR / "quant_config.json"))
    print(f"  Copied quant_config.json from ViT donor")

# --- 3. Load tensors ---
print("[3/5] Loading tensors...")

# Load ViT donor tensors (AQLM model, 3 shards)
vit_tensors = {}
for sf in sorted(VIT_DONOR.glob("model*.safetensors")):
    ts = load_file(str(sf), device="cpu")
    vit_tensors.update(ts)
print(f"  ViT donor: {len(vit_tensors)} keys total")
vit_keys = [k for k in vit_tensors if k.startswith("vision_model.")]
print(f"    vision_model.* keys: {len(vit_keys)}")

# Load LLM donor tensors (miniViT_distilled, 1 shard)
llm_tensors = {}
for sf in sorted(LLM_DONOR.glob("model*.safetensors")):
    ts = load_file(str(sf), device="cpu")
    llm_tensors.update(ts)
print(f"  LLM donor: {len(llm_tensors)} keys total")
llm_keys = [k for k in llm_tensors if not k.startswith("vision_model.")]
vit_keys_to_drop = [k for k in llm_tensors if k.startswith("vision_model.")]
print(f"    non-vision_model keys to keep: {len(llm_keys)}")
print(f"    vision_model keys to drop: {len(vit_keys_to_drop)}")

# --- 4. Merge ---
print("[4/5] Merging tensors...")
merged = OrderedDict()

# Add non-ViT keys from LLM donor
for k in sorted(llm_keys):
    merged[k] = llm_tensors[k]

# Add ViT keys from ViT donor
for k in sorted(vit_keys):
    merged[k] = vit_tensors[k]

total_keys = len(merged)
total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
print(f"  Merged: {total_keys} keys, {total_bytes/1024/1024:.1f} MB ({total_bytes/1024/1024/1024:.2f} GB)")

# --- 5. Save ---
print("[5/5] Saving to safetensors...")

# Check if we need sharding (>5GB per shard is the safetensors limit, but we should be fine)
# Generate metadata dict
metadata = {"format": "pt"}
save_file(merged, str(OUTPUT_DIR / "model.safetensors"), metadata=metadata)

# Generate index.json
print("  Generating index.json...")
weight_map = OrderedDict()
for k in merged:
    weight_map[k] = "model.safetensors"

index = {
    "metadata": {"total_size": total_bytes},
    "weight_map": weight_map,
}
with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

# Verify
file_size = (OUTPUT_DIR / "model.safetensors").stat().st_size
print(f"  model.safetensors: {file_size/1024/1024:.1f} MB")
print(f"\nDone! Model saved to: {OUTPUT_DIR}")

# Quick verification: check key counts match
print(f"\n=== Verification ===")
print(f"  ViT keys (from AQLM):        {len(vit_keys)}")
print(f"  LLM keys (from miniViT):     {len(llm_keys)}")
print(f"  Total merged keys:           {total_keys}")
print(f"  ViT blocks with full weights: {len(set(int(re.search(r'blocks\.(\d+)', k).group(1)) for k in merged if re.search(r'vision_model.blocks\.(\d+)', k)))}")
