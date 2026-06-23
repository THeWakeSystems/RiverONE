#!/usr/bin/env python3
"""Fix PV-tuned model dtypes for AQLM inference.

Problem: train_pv_tuning.py saves codes as I32 (int32), codebooks as F32 (float32).
AQLM inference engine (riverone_backend_aqlm.py) expects codes=int16 and codebooks=BF16.

This script loads the model, casts to correct dtypes, and re-saves.
"""
import json
import safetensors.torch as st
from pathlib import Path
import torch

MODEL_DIR = Path(__file__).resolve().parent.parent / "weights" / "pv_mlponly_v2"
# Override with command-line arg or edit this path for your environment
import sys
if len(sys.argv) > 1:
    MODEL_DIR = Path(sys.argv[1])
SF_IN = MODEL_DIR / "model.safetensors"
SF_BAK = MODEL_DIR / "model.safetensors.bak"  # backup original
SF_OUT = MODEL_DIR / "model.safetensors"

print(f"Loading {SF_IN}...")
tensors = {}
with st.safe_open(str(SF_IN), framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(f"Loaded {len(tensors)} tensors")

# Track changes
n_codes_fixed = 0
n_cb_fixed = 0
n_scales_fixed = 0

for key, t in tensors.items():
    if key == "__metadata__":
        continue
    
    dtype_str = str(t.dtype).replace("torch.", "")
    
    if "codes" in key:
        if t.dtype == torch.int32:
            # I32 -> I16 (nbits=16 supports indices 0-65535)
            tensors[key] = t.to(torch.int16)
            n_codes_fixed += 1
            print(f"  CAST: {key}: {dtype_str}({list(t.shape)}) -> I16")
        elif t.dtype != torch.int16:
            print(f"  WARN: unexpected codes dtype: {dtype_str} for {key}")
            
    elif "codebooks" in key:
        if t.dtype == torch.float32:
            tensors[key] = t.to(torch.bfloat16)
            n_cb_fixed += 1
            print(f"  CAST: {key}: {dtype_str}({list(t.shape)}) -> BF16")
        elif t.dtype != torch.bfloat16:
            print(f"  WARN: unexpected codebooks dtype: {dtype_str} for {key}")
            
    elif "scales" in key:
        if t.dtype == torch.float32:
            tensors[key] = t.to(torch.bfloat16)
            n_scales_fixed += 1
            # only print first few
            if n_scales_fixed <= 3:
                print(f"  CAST: {key}: {dtype_str}({list(t.shape)}) -> BF16")
        elif t.dtype != torch.bfloat16:
            print(f"  WARN: unexpected scales dtype: {dtype_str} for {key}")

print(f"\nFixed: {n_codes_fixed} codes (I32→I16), {n_cb_fixed} codebooks (F32→BF16), {n_scales_fixed} scales (F32→BF16)")

# Backup original
if not SF_BAK.exists():
    SF_IN.rename(SF_BAK)
    print(f"Backup: {SF_IN} -> {SF_BAK}")

# Save fixed model
print(f"Saving fixed model to {SF_OUT}...")
st.save_file(tensors, str(SF_OUT))
print(f"Done. New size: {SF_OUT.stat().st_size / (1024**3):.2f} GB")
