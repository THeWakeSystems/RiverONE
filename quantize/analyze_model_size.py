#!/usr/bin/env python3
"""分析 32L MLP-only AQLM 模型的参数和权重大小明细"""
import json
from collections import defaultdict
from safetensors import safe_open

INDEX = "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-MLPonly/model.safetensors.index.json"
SHARDS = [
    "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-MLPonly/model-00001-of-00003.safetensors",
    "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-MLPonly/model-00002-of-00003.safetensors",
    "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-MLPonly/model-00003-of-00003.safetensors",
]

with open(INDEX) as f:
    idx = json.load(f)
wm = idx["weight_map"]

aqlm_keys = [k for k in wm if k.endswith((".codebooks", ".codes", ".scales"))]
bf16_keys = [k for k in wm if k not in aqlm_keys]

shape_map = {}
for sf in SHARDS:
    with safe_open(sf, framework="pt") as f:
        for key in f.keys():
            shape_map[key] = f.get_tensor(key).shape

# Count bytes
bf16_bytes = 0
bf16_params = 0
for key in bf16_keys:
    if key in shape_map:
        shape = shape_map[key]
        nelem = 1
        for s in shape:
            nelem *= s
        bf16_params += nelem
        bf16_bytes += nelem * 2

aqlm_total_bytes = 0
mlp_aqlm_bytes = 0
for key in aqlm_keys:
    if key in shape_map:
        shape = shape_map[key]
        nelem = 1
        for s in shape:
            nelem *= s
        b = nelem if key.endswith(".codes") else nelem * 2
        aqlm_total_bytes += b
        if "mlp" in key and "layers" in key:
            mlp_aqlm_bytes += b

def categorize(key):
    if "vision_model" in key:
        return "ViT 视觉编码器"
    if "language_model.model.embed_tokens" in key:
        return "LLM Embedding"
    if "language_model.lm_head" in key:
        return "LLM LM Head"
    if "language_model.model.norm" in key:
        return "LLM Final Norm"
    if "language_model.model.layers" in key:
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "layers":
                ln = int(parts[i+1])
                break
        if "self_attn" in key:
            return "LLM Attn bf16 (L0-L3)" if ln < 4 else "LLM Attn bf16 (L4-L35)"
        if "mlp" in key:
            return "LLM MLP bf16 (L0-L3)" if ln < 4 else "LLM MLP AQLM (L4-L35)"
        if "layernorm" in key:
            return "LLM LayerNorm"
        if "q_norm" in key or "k_norm" in key:
            return "LLM QK Norm"
        return "LLM Other"
    if "mlp1" in key:
        return "mlp1 投影层"
    return "Other"

cat_bytes = defaultdict(int)
cat_params = defaultdict(int)

for key in bf16_keys:
    cat = categorize(key)
    if key in shape_map:
        shape = shape_map[key]
        nelem = 1
        for s in shape: nelem *= s
        cat_params[cat] += nelem
        cat_bytes[cat] += nelem * 2

cat_bytes["LLM MLP AQLM (L4-L35)"] += mlp_aqlm_bytes

# AQLM per-projection detail
cb_key = "language_model.model.layers.4.mlp.gate_proj.codebooks"
codes_key = "language_model.model.layers.4.mlp.gate_proj.codes"
scales_key = "language_model.model.layers.4.mlp.gate_proj.scales"
cb_shape = list(shape_map[cb_key])
codes_shape = list(shape_map[codes_key])
scales_shape = list(shape_map[scales_key])

def nelem(shape):
    p = 1
    for s in shape: p *= s
    return p

cb_bytes_per = nelem(cb_shape) * 2
codes_bytes_per = nelem(codes_shape)
scales_bytes_per = nelem(scales_shape) * 2
per_proj = cb_bytes_per + codes_bytes_per + scales_bytes_per

# Original MLP
orig_mlp = 32 * 3 * 9728 * 2560 * 2
total_bytes = bf16_bytes + aqlm_total_bytes
implicit = 32 * 3 * 9728 * 2560
total_orig = 9750000000

print("=" * 64)
print(" 32L MLP-only AQLM — 参数 & 权重大小明细")
print("=" * 64)
print()
print(f"  Total keys: {len(wm)}  (bf16: {len(bf16_keys)}, AQLM: {len(aqlm_keys)})")
print()
print(f"  显式 bf16 参数:  {bf16_params:>12,}  ({bf16_params/1e9:.2f}B)")
print(f"  隐式 AQLM 参数:  {implicit:>12,}  ({implicit/1e9:.2f}B) — 运行时码书重建")
print(f"  模型总参数量:    {bf16_params + implicit:>12,}  ({(bf16_params+implicit)/1e9:.2f}B)")
print(f"  (原始模型: ~4.88B — 参数量不变, 仅存储格式变化)")
print()
print(f"  权重大小:       {total_bytes/1e9:.2f} GB  (vs 源 {total_orig/1e9:.2f} GB)")
print(f"  整体压缩比:     {total_orig/total_bytes:.2f}x")
print(f"  整体节省:       {(1-total_bytes/total_orig)*100:.1f}%")
print()
print(f"  ── AQLM 每组投影结构 (1 projection) ──")
print(f"    codebooks  {cb_shape}  = {cb_bytes_per/1e6:.3f} MB  (bf16)")
print(f"    codes      {codes_shape}  = {codes_bytes_per/1e6:.3f} MB  (int8)")
print(f"    scales     {scales_shape}  = {scales_bytes_per/1e6:.3f} MB  (bf16)")
print(f"    单组合计   = {per_proj/1e6:.3f} MB")
print()
print(f"  ── 按组件权重大小 ──")
print(f"  {'组件':<30} {'参数数':>12} {'大小':>10} {'占比':>7}")
print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*7}")

for cat in sorted(cat_bytes.keys(), key=lambda c: cat_bytes[c], reverse=True):
    cb = cat_bytes[cat]
    cp = cat_params[cat]
    if cb > 0:
        pct = cb / total_bytes * 100
        print(f"  {cat:<30} {cp:>12,} {cb/1e6:>8.1f} MB {pct:>5.1f}%")

print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*7}")
print(f"  {'合计':<30} {bf16_params:>12,} {total_bytes/1e9:>8.2f} GB")

print()
print(f"  ── MLP L4-L35 压缩明细 ──")
print(f"    原始 bf16:     {orig_mlp/1e6:>8.1f} MB  (32层 x 3投影 x 9728x2560 x 2B)")
print(f"    AQLM 量化:     {mlp_aqlm_bytes/1e6:>8.1f} MB")
print(f"    压缩比:        {orig_mlp/mlp_aqlm_bytes:.1f}x")
print(f"    等效 bit/param: {16/(orig_mlp/mlp_aqlm_bytes):.2f} bpw")
print()
print(f"    AQLM 96组 = {per_proj*96/1e6:.1f} MB (96 projections x {per_proj/1e6:.3f} MB)")
print(f"    实际存储: {mlp_aqlm_bytes/1e6:.1f} MB")
