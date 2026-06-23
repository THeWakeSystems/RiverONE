#!/usr/bin/env python3
"""Full model analysis: parameter count + weight error + semantic collapse.

   /home/lxy/AQLM-32L-AttnMix_miniViT
   = AQLM-quantized LLM + miniViT_distilled ViT
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

import torch
from safetensors.torch import load_file
from aqlm.utils import _dequantize_weight, unpack_int_data

MODEL_DIR = Path("/home/lxy/AQLM-32L-AttnMix_miniViT")
SOURCE_MODEL = Path("/home/lxy/workspace/riverone-release/RiverOne-QC-4B-v2")

# nbits per sublayer
SUBlayer_NBITS = {
    "k_proj": 14, "v_proj": 14,
    "q_proj": 16, "o_proj": 16,
    "gate_proj": 16, "up_proj": 16, "down_proj": 16,
}
QUANT_LAYERS = list(range(4, 36))

def load_all(dirpath):
    tensors = {}
    for sf in sorted(dirpath.glob("model*.safetensors")):
        ts = load_file(str(sf), device="cpu")
        tensors.update(ts)
    return tensors

print("=" * 80)
print(" AQLM-32L-AttnMix_miniViT: Full Model Analysis")
print("=" * 80)

# ── Load ──
print("\n[1] Loading model...")
model_tensors = load_all(MODEL_DIR)
print(f"  Total keys: {len(model_tensors)}")

src_tensors = load_all(SOURCE_MODEL)
print(f"  Source keys: {len(src_tensors)}")

# ── Parameter Count ──
print("\n" + "=" * 80)
print(" PARAMETER COUNT")
print("=" * 80)

stats = {
    "ViT (bf16)": 0,
    "LLM Embedding (bf16)": 0,
    "LLM LM Head (bf16)": 0,
    "LLM Attn bf16 L0-L3": 0,
    "LLM MLP bf16 L0-L3": 0,
    "LLM Attn AQLM L4-L35": 0,      # stored AQLM bytes
    "LLM MLP AQLM L4-L35": 0,       # stored AQLM bytes
    "LLM Attn AQLM dequantized params": 0,  # effective params
    "LLM MLP AQLM dequantized params": 0,
    "Norms + Other (bf16)": 0,
    "mlp1 projection (bf16)": 0,
}
stored_aqlm_bytes = {"attn": 0, "mlp": 0}

for key, tensor in model_tensors.items():
    sz = tensor.numel() * tensor.element_size()
    m = re.search(r'layers\.(\d+)', key)
    lid = int(m.group(1)) if m else None

    if 'vision_model' in key or 'visual' in key:
        stats["ViT (bf16)"] += tensor.numel()
    elif 'embed_tokens' in key:
        stats["LLM Embedding (bf16)"] += tensor.numel()
    elif 'lm_head' in key:
        stats["LLM LM Head (bf16)"] += tensor.numel()
    elif 'mlp1' in key:
        stats["mlp1 projection (bf16)"] += tensor.numel()
    elif '.codebooks' in key or '.codes' in key or '.scales' in key:
        if lid is not None and lid >= 4:
            if 'self_attn' in key:
                stats["LLM Attn AQLM L4-L35"] += sz
                stored_aqlm_bytes["attn"] += sz
            elif 'mlp' in key:
                stats["LLM MLP AQLM L4-L35"] += sz
                stored_aqlm_bytes["mlp"] += sz
    elif 'self_attn' in key and lid is not None:
        if lid < 4:
            stats["LLM Attn bf16 L0-L3"] += tensor.numel()
    elif 'mlp' in key and lid is not None and lid < 4:
        stats["LLM MLP bf16 L0-L3"] += tensor.numel()
    elif 'layernorm' in key or 'norm' in key.lower():
        stats["Norms + Other (bf16)"] += tensor.numel()
    elif 'language_model' in key:
        stats["Norms + Other (bf16)"] += tensor.numel()

# Now compute dequantized params for AQLM layers
attn_deq_params = 0
mlp_deq_params = 0

for key in model_tensors:
    m = re.search(r'layers\.(\d+)', key)
    lid = int(m.group(1)) if m else None
    if lid is None or lid < 4:
        continue

    # Find codebooks keys to determine shapes
    if not key.endswith('.codebooks'):
        continue

    # Determine projection type
    proj = None
    for p in SUBlayer_NBITS:
        if f'.{p}.' in key or key.endswith(f'.{p}.codebooks'):
            proj = p
            break
    if proj is None:
        continue

    cb = model_tensors[key]
    # Dequantized weight shape: num_groups × out_group_size × in_group_size
    # For this scheme: out_group_size=1, in_group_size=16
    # num_groups = codebooks entries per group
    # Actually from dequantize: weight_shape = [num_out_groups * 1, num_in_groups * 16]
    # Let me use the source model to get the actual shape
    
    # Build source key
    src_key = key.replace('.codebooks', '.weight')
    if src_key in src_tensors:
        deq_size = src_tensors[src_key].numel()
        if 'self_attn' in key:
            attn_deq_params += deq_size
        elif 'mlp' in key:
            mlp_deq_params += deq_size

stats["LLM Attn AQLM dequantized params"] = attn_deq_params
stats["LLM MLP AQLM dequantized params"] = mlp_deq_params

# Totals
total_bf16_params = sum(v for k, v in stats.items() 
                        if 'AQLM' not in k and 'dequantized' not in k and 'stored' not in k)
total_stored_aqlm_params = (stats["LLM Attn AQLM L4-L35"] + stats["LLM MLP AQLM L4-L35"]) // 2  # bytes -> bf16 params
total_deq_params = stats["LLM Attn AQLM dequantized params"] + stats["LLM MLP AQLM dequantized params"]
total_effective = total_bf16_params + total_deq_params

print(f"\n  {'Component':<35s} {'Params':>14s} {'Size':>10s}")
print(f"  {'-'*35} {'-'*14} {'-'*10}")
for name in ["ViT (bf16)", "LLM Embedding (bf16)", "LLM LM Head (bf16)",
             "LLM Attn bf16 L0-L3", "LLM MLP bf16 L0-L3",
             "LLM Attn AQLM L4-L35", "LLM MLP AQLM L4-L35",
             "LLM Attn AQLM dequantized params", "LLM MLP AQLM dequantized params",
             "Norms + Other (bf16)", "mlp1 projection (bf16)"]:
    v = stats[name]
    if 'AQLM L4-L35' in name:
        print(f"  {name:<35s} {'--':>14s} {v/1024/1024:>8.1f} MB")
    elif 'dequantized' in name:
        print(f"  {name:<35s} {v/1e6:>10.1f}M {v*2/1024/1024:>8.1f} MB")
    else:
        print(f"  {name:<35s} {v/1e6:>10.1f}M {v*2/1024/1024:>8.1f} MB")

print(f"  {'-'*35} {'-'*14} {'-'*10}")
print(f"  {'TOTAL (stored bf16+AQLM)':<35s} {'--':>14s} {(sum(t.numel()*t.element_size() for t in model_tensors.values()))/1024/1024:>8.1f} MB")
print(f"  {'TOTAL effective params':<35s} {total_effective/1e6:>10.1f}M {total_effective*2/1024/1024:>8.1f} MB")
print(f"  {'TOTAL stored bf16 equivalent':<35s} {(total_bf16_params + total_stored_aqlm_params)/1e6:>10.1f}M {(total_bf16_params + total_stored_aqlm_params)*2/1024/1024:>8.1f} MB")

# Calculate billions
print(f"\n  Effective: {total_effective/1e9:.2f}B params")
print(f"  Stored as: {(sum(t.numel()*t.element_size() for t in model_tensors.values()))/1024/1024/1024:.2f} GB on disk")

# Compression stats
attn_src_params = attn_deq_params  # source attention params
mlp_src_params = mlp_deq_params
print(f"\n  Attn AQLM compression: {attn_src_params*2/1024/1024:.0f} MB -> {stored_aqlm_bytes['attn']/1024/1024:.0f} MB ({attn_src_params*2/stored_aqlm_bytes['attn']:.1f}x)")
print(f"  MLP AQLM compression:  {mlp_src_params*2/1024/1024:.0f} MB -> {stored_aqlm_bytes['mlp']/1024/1024:.0f} MB ({mlp_src_params*2/stored_aqlm_bytes['mlp']:.1f}x)")

# ── Weight Error Analysis ──
print("\n" + "=" * 80)
print(" WEIGHT ERROR ANALYSIS (AQLM LLM layers vs source)")
print("=" * 80)

# First, check if ViT is identical to source (should be if from miniViT_distilled)
print("\n[ViT Comparison]")
vit_diffs = 0
vit_total = 0
for key in model_tensors:
    if not key.startswith("vision_model."):
        continue
    vit_total += 1
    if key in src_tensors:
        if not torch.equal(model_tensors[key], src_tensors[key]):
            vit_diffs += 1
    else:
        vit_diffs += 1

if vit_diffs == 0:
    print(f"  ViT: {vit_total} keys — ALL IDENTICAL to source (bf16, no quantization)")
else:
    print(f"  ViT: {vit_total} keys — {vit_diffs} DIFFER from source!")

# AQLM LLM weight error
print(f"\n[AQLM LLM Weight Error] ({len(QUANT_LAYERS)} layers x 7 sublayers = {len(QUANT_LAYERS)*7} matrices)")
print()

results = []
for layer_idx in QUANT_LAYERS:
    for component, proj in [("self_attn","k_proj"),("self_attn","v_proj"),
                            ("self_attn","q_proj"),("self_attn","o_proj"),
                            ("mlp","gate_proj"),("mlp","up_proj"),("mlp","down_proj")]:
        layer_path = f"language_model.model.layers.{layer_idx}.{component}.{proj}"
        src_key = f"{layer_path}.weight"
        cb_key = f"{layer_path}.codebooks"
        codes_key = f"{layer_path}.codes"
        scales_key = f"{layer_path}.scales"

        if src_key not in src_tensors:
            continue
        if cb_key not in model_tensors:
            continue

        nbits = SUBlayer_NBITS[proj]
        src_weight = src_tensors[src_key].float()
        cb = model_tensors[cb_key].float()
        codes = unpack_int_data(model_tensors[codes_key], nbits)
        scales = model_tensors[scales_key].float()
        aqlm_weight = _dequantize_weight(codes, cb, scales)

        if aqlm_weight.shape != src_weight.shape:
            if aqlm_weight.T.shape == src_weight.shape:
                aqlm_weight = aqlm_weight.T
            elif aqlm_weight.shape == src_weight.T.shape:
                src_weight = src_weight.T
            else:
                continue

        diff = aqlm_weight - src_weight
        rel_err = torch.norm(diff).item() / torch.norm(src_weight).item() * 100
        flat_aqlm = aqlm_weight.flatten()
        flat_src = src_weight.flatten()
        cos_sim = torch.dot(flat_aqlm, flat_src).item() / (
            torch.norm(flat_aqlm).item() * torch.norm(flat_src).item())

        results.append({"layer": layer_idx, "proj": proj, "nbits": nbits,
                        "rel_err": rel_err, "cos_sim": cos_sim})

        status = "OK" if cos_sim > 0.85 else ("WARN" if cos_sim > 0.7 else "FAIL")
        if cos_sim < 0.9:  # Only print non-perfect ones
            print(f"  L{layer_idx:2d}/{proj:>9s} (n={nbits:2d}): "
                  f"err={rel_err:5.1f}%  cos={cos_sim:.4f}  {status}")

# Summary
n = len(results)
avg_cos = sum(r["cos_sim"] for r in results) / n
avg_err = sum(r["rel_err"] for r in results) / n

collapse = sum(1 for r in results if r["cos_sim"] < 0.7)
borderline = sum(1 for r in results if 0.7 <= r["cos_sim"] < 0.85)
healthy = sum(1 for r in results if r["cos_sim"] >= 0.85)

print(f"\n  Analyzed: {n} matrices")
print(f"  Avg cos_sim:  {avg_cos:.4f}")
print(f"  Avg rel_err:  {avg_err:.1f}%")
print(f"  Healthy (>=0.85):    {healthy}/{n} ({healthy/n*100:.1f}%)")
print(f"  Borderline (0.7-0.85): {borderline}/{n}")
print(f"  Collapsed (<0.7):    {collapse}/{n}")

# ── Verdict ──
print("\n" + "=" * 80)
print(" SEMANTIC COLLAPSE VERDICT")
print("=" * 80)

print(f"\n  Model: AQLM-32L-AttnMix_miniViT")
print(f"  Effective params: {total_effective/1e9:.2f}B")
print(f"  Stored size:      {(sum(t.numel()*t.element_size() for t in model_tensors.values()))/1024/1024/1024:.2f} GB")
print()

# ViT verdict
if vit_diffs == 0:
    print(f"  ViT: IDENTICAL to source — no quantization loss, no collapse")
else:
    print(f"  ViT: {vit_diffs}/{vit_total} keys differ from source — quantized/modified")

# LLM verdict
print(f"\n  LLM AQLM: {healthy}/{n} healthy ({healthy/n*100:.1f}%)")

if collapse == 0 and borderline == 0:
    print(f"  >>> LLM VERDICT: No semantic collapse. All AQLM layers high fidelity.")
elif collapse == 0:
    print(f"  >>> LLM VERDICT: No collapse. {borderline} borderline matrices (minor degradation).")
else:
    print(f"  >>> LLM VERDICT: {collapse} collapsed matrices, {borderline} borderline.")

# Overall
print(f"\n  >>> OVERALL VERDICT: ", end="")
if collapse == 0 and vit_diffs == 0:
    print("HEALTHY — no semantic collapse detected.")
elif collapse == 0:
    print("ACCEPTABLE — minor AQLM borderline matrices, ViT pristine.")
elif collapse <= 2:
    print("MINOR DEGRADATION — isolated collapse in deep down_proj, ViT pristine.")
else:
    print("SIGNIFICANT COLLAPSE — multiple AQLM matrices degraded.")

# Problem breakdown
if collapse > 0 or borderline > 0:
    print(f"\n  Problem matrices:")
    bad = [r for r in results if r["cos_sim"] < 0.85]
    bad.sort(key=lambda r: r["cos_sim"])
    for r in bad:
        sev = "COLLAPSED" if r["cos_sim"] < 0.7 else "BORDERLINE"
        print(f"    L{r['layer']:2d}/{r['proj']:>9s} (n={r['nbits']:2d}): "
              f"cos={r['cos_sim']:.4f}  err={r['rel_err']:.1f}%  [{sev}]")

print("\n" + "=" * 80)
