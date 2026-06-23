#!/usr/bin/env python3
"""Reverse ViT Swap: AQLM quantized LLM + miniViT_distilled ViT.

Source models:
  - LLM donor: AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-Attn14MLP16 (quantized LLM)
  - ViT donor: /home/lxy/miniViT_distilled/ (miniViT with weight sharing + distilled transforms)

Output: AQLM-quantized LLM + miniViT'd ViT.
Existing model directories are NOT modified.
"""
from __future__ import annotations
import json, shutil, re
from pathlib import Path
from collections import OrderedDict

from safetensors.torch import load_file, save_file

# --- Config ---
LLM_DONOR = Path("/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-Attn14MLP16")
VIT_DONOR = Path("/home/lxy/miniViT_distilled")
OUTPUT_DIR = Path("/home/lxy/AQLM-32L-AttnMix_miniViT")

# --- 1. Create output directory ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[1/5] Output directory: {OUTPUT_DIR}")

# --- 2. Copy config files ---
print("[2/5] Copying config files...")
# From LLM donor (AQLM model): config, tokenizer, model code, quant_config
COPY_FROM_LLM = [
    "config.json", "generation_config.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
    "configuration_riverone_qc.py", "modeling_riverone_qc.py", "modeling_ising_vit.py",
    "conversation.py", "preprocessor_config.json", "processor_config.json",
    "chat_template.jinja", "video_preprocessor_config.json",
    "quant_config.json",
]
for fname in COPY_FROM_LLM:
    src = LLM_DONOR / fname
    if src.exists():
        shutil.copy2(str(src), str(OUTPUT_DIR / fname))

# miniViT_config.json from ViT donor (reflects weight sharing + distilled transforms)
mini_src = VIT_DONOR / "miniViT_config.json"
if mini_src.exists():
    shutil.copy2(str(mini_src), str(OUTPUT_DIR / "miniViT_config.json"))
    print(f"  miniViT_config.json from ViT donor (weight_sharing=true)")

print("  Config files copied")

# --- 3. Load tensors ---
print("[3/5] Loading tensors...")

# Load LLM donor (AQLM model)
llm_tensors = {}
for sf in sorted(LLM_DONOR.glob("model*.safetensors")):
    ts = load_file(str(sf), device="cpu")
    llm_tensors.update(ts)
print(f"  LLM donor (AQLM): {len(llm_tensors)} keys total")

# Split LLM donor into LLM vs ViT
llm_donor_llm = [k for k in llm_tensors if not k.startswith("vision_model.")]
llm_donor_vit = [k for k in llm_tensors if k.startswith("vision_model.")]
print(f"    -> LLM keys to keep: {len(llm_donor_llm)}")
print(f"    -> ViT keys to drop: {len(llm_donor_vit)}")

# Count AQLM vs bf16 in LLM donor
aqlm_count = sum(1 for k in llm_donor_llm if any(x in k for x in ['.codebooks', '.codes', '.scales']))
bf16_count = len(llm_donor_llm) - aqlm_count
print(f"       AQLM quantized: {aqlm_count}, bf16: {bf16_count}")

# Load ViT donor (miniViT_distilled)
vit_tensors = {}
for sf in sorted(VIT_DONOR.glob("model*.safetensors")):
    ts = load_file(str(sf), device="cpu")
    vit_tensors.update(ts)
print(f"  ViT donor (miniViT_distilled): {len(vit_tensors)} keys total")

vit_donor_vit = [k for k in vit_tensors if k.startswith("vision_model.")]
vit_donor_llm = [k for k in vit_tensors if not k.startswith("vision_model.")]
print(f"    -> ViT keys to keep: {len(vit_donor_vit)}")
print(f"    -> LLM keys to drop: {len(vit_donor_llm)}")

# --- 4. Merge ---
print("[4/5] Merging tensors...")
merged = OrderedDict()

# LLM from AQLM model (includes mlp1 projection layer)
for k in sorted(llm_donor_llm):
    merged[k] = llm_tensors[k]

# ViT from miniViT_distilled
for k in sorted(vit_donor_vit):
    merged[k] = vit_tensors[k]

total_keys = len(merged)
total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
print(f"  Merged: {total_keys} keys, {total_bytes/1024/1024:.1f} MB ({total_bytes/1024/1024/1024:.2f} GB)")

# --- 5. Save ---
print("[5/5] Saving...")

# Generate metadata with AQLM format indicator if quantized
metadata = {"format": "pt"}
save_file(merged, str(OUTPUT_DIR / "model.safetensors"), metadata=metadata)

# Generate index.json
weight_map = OrderedDict()
for k in merged:
    weight_map[k] = "model.safetensors"

index = {
    "metadata": {"total_size": total_bytes},
    "weight_map": weight_map,
}
with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

file_size = (OUTPUT_DIR / "model.safetensors").stat().st_size
print(f"  model.safetensors: {file_size/1024/1024:.1f} MB")

print(f"\nDone! Model saved to: {OUTPUT_DIR}")

# --- Verification ---
print(f"\n=== Verification ===")
print(f"  LLM keys (from AQLM):     {len(llm_donor_llm)}  (AQLM: {aqlm_count}, bf16: {bf16_count})")
print(f"  ViT keys (from miniViT):  {len(vit_donor_vit)}")
print(f"  Total merged:             {total_keys}")
# Check ViT structure
blocks = set()
for k in merged:
    m = re.search(r'vision_model.blocks.(\d+)', k)
    if m: blocks.add(int(m.group(1)))
print(f"  ViT blocks:               {len(blocks)} (0-{max(blocks)})")
transform_count = sum(1 for k in merged if 'transform' in k.lower())
print(f"  Transform keys present:   {transform_count} (miniViT weight sharing)")
block24_keys = sum(1 for k in merged if 'vision_model.blocks.24' in k)
print(f"  Block 24 keys:            {block24_keys} (shared ViT should be < 12)")
