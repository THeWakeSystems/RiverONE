#!/usr/bin/env python3
"""分析 AQLM-32L-AttnMix_miniViT 模型的参数和权重大小明细。

   LLM: AQLM 2x16 量化 (k/v=n14, q/o/mlp=n16), L4-L35
   ViT: miniViT_distilled (权重共享 + 蒸馏 transform)
"""
import json, re
from collections import defaultdict
from safetensors import safe_open

MODEL_DIR = "/home/lxy/AQLM-32L-AttnMix_miniViT"
SAFETENSORS = [f"{MODEL_DIR}/model.safetensors"]
INDEX = f"{MODEL_DIR}/model.safetensors.index.json"

with open(INDEX) as f:
    idx = json.load(f)
wm = idx["weight_map"]

aqlm_keys = [k for k in wm if k.endswith((".codebooks", ".codes", ".scales"))]
bf16_keys = [k for k in wm if k not in aqlm_keys]

shape_map = {}
for sf in SAFETENSORS:
    with safe_open(sf, framework="pt") as f:
        for key in f.keys():
            shape_map[key] = f.get_tensor(key).shape

# ── Count bytes ──
bf16_bytes = 0
bf16_params = 0
for key in bf16_keys:
    if key in shape_map:
        nelem = 1
        for s in shape_map[key]:
            nelem *= s
        bf16_params += nelem
        bf16_bytes += nelem * 2

aqlm_total_bytes = 0
attn_aqlm_bytes = 0
mlp_aqlm_bytes = 0
for key in aqlm_keys:
    if key in shape_map:
        nelem = 1
        for s in shape_map[key]:
            nelem *= s
        b = nelem if key.endswith(".codes") else nelem * 2
        aqlm_total_bytes += b
        if "self_attn" in key:
            attn_aqlm_bytes += b
        elif "mlp" in key:
            mlp_aqlm_bytes += b

# ── Categorize ──
def categorize(key):
    if "vision_model" in key:
        return "ViT 视觉编码器 (miniViT)"
    if "language_model.model.embed_tokens" in key:
        return "LLM Embedding"
    if "language_model.lm_head" in key:
        return "LLM LM Head"
    if "language_model.model.norm" in key:
        return "LLM Final Norm"
    if "language_model.model.layers" in key:
        m = re.search(r"layers\.(\d+)", key)
        ln = int(m.group(1)) if m else -1
        if "self_attn" in key:
            if ".codebooks" in key or ".codes" in key or ".scales" in key:
                return "LLM Attn AQLM (L4-L35)"
            return "LLM Attn bf16 (L0-L3)" if ln < 4 else "LLM Attn bf16"
        if "mlp" in key:
            if ".codebooks" in key or ".codes" in key or ".scales" in key:
                return "LLM MLP AQLM (L4-L35)"
            return "LLM MLP bf16 (L0-L3)" if ln < 4 else "LLM MLP bf16"
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
        nelem = 1
        for s in shape_map[key]: nelem *= s
        cat_params[cat] += nelem
        cat_bytes[cat] += nelem * 2

cat_bytes["LLM MLP AQLM (L4-L35)"] += mlp_aqlm_bytes
cat_bytes["LLM Attn AQLM (L4-L35)"] += attn_aqlm_bytes

# ── AQLM per-projection detail ──
# Attn: q_proj/o_proj = [2560,2560], k_proj/v_proj = [1024,2560] (nbits=16 or 14)
# MLP: gate/up = [9728,2560], down = [2560,9728] (nbits=16)

def nelem(shape):
    p = 1
    for s in shape: p *= s
    return p

def show_proj(label, cb_key, codes_key, scales_key):
    cb_shape = list(shape_map[cb_key])
    codes_shape = list(shape_map[codes_key])
    scales_shape = list(shape_map[scales_key])
    cb_b = nelem(cb_shape) * 2
    codes_b = nelem(codes_shape)
    scales_b = nelem(scales_shape) * 2
    total = cb_b + codes_b + scales_b
    return cb_shape, codes_shape, scales_shape, cb_b, codes_b, scales_b, total

# MLP projection
mlp_proj = show_proj(
    "MLP", 
    "language_model.model.layers.4.mlp.gate_proj.codebooks",
    "language_model.model.layers.4.mlp.gate_proj.codes",
    "language_model.model.layers.4.mlp.gate_proj.scales",
)

# Attn q_proj (n=16, [2560,2560])
attn_q_proj = show_proj(
    "Attn q_proj",
    "language_model.model.layers.4.self_attn.q_proj.codebooks",
    "language_model.model.layers.4.self_attn.q_proj.codes",
    "language_model.model.layers.4.self_attn.q_proj.scales",
)

# Attn k_proj (n=14, [1024,2560])
attn_k_proj = show_proj(
    "Attn k_proj",
    "language_model.model.layers.4.self_attn.k_proj.codebooks",
    "language_model.model.layers.4.self_attn.k_proj.codes",
    "language_model.model.layers.4.self_attn.k_proj.scales",
)

# ── Dequantized param count ──
# MLP: 32 layers x 3 proj x 9728 x 2560 (gate+up) or 2560 x 9728 (down)
mlp_deq_per_layer = (9728 * 2560 * 2 + 2560 * 9728)  # gate+up+down
mlp_deq_total = 32 * mlp_deq_per_layer
# Attn: 32 layers x [q:2560x2560, k:1024x2560, v:1024x2560, o:2560x2560]
attn_deq_per_layer = 2560*2560 + 1024*2560 + 1024*2560 + 2560*2560
attn_deq_total = 32 * attn_deq_per_layer
total_deq = bf16_params + mlp_deq_total + attn_deq_total

# ── Print ──
print("=" * 70)
print(" AQLM-32L-AttnMix_miniViT — 参数 & 权重大小明细")
print("=" * 70)
print()
print(f"  方案: 2×16 AttnMix (k/v=n14, q/o/mlp=n16), L4-L35, 32层")
print(f"  ViT: miniViT_distilled (权重共享 + 蒸馏)")
print()
print(f"  Total keys: {len(wm)}  (bf16: {len(bf16_keys)}, AQLM: {len(aqlm_keys)})")
print(f"    AQLM attn: {sum(1 for k in aqlm_keys if 'self_attn' in k)}  "
      f"({sum(1 for k in aqlm_keys if 'self_attn' in k)//3} 组)")
print(f"    AQLM mlp:  {sum(1 for k in aqlm_keys if 'mlp' in k)}  "
      f"({sum(1 for k in aqlm_keys if 'mlp' in k)//3} 组)")
print()

print(f"  ── 参数量 ──")
print(f"  显式 bf16 参数:    {bf16_params:>12,}  ({bf16_params/1e9:.2f}B)")
print(f"  AQLM 等效参数:")
print(f"    MLP (L4-L35):    {mlp_deq_total:>12,}  ({mlp_deq_total/1e9:.2f}B)")
print(f"    Attn (L4-L35):   {attn_deq_total:>12,}  ({attn_deq_total/1e9:.2f}B)")
print(f"  模型总参数量:      {total_deq:>12,}  ({total_deq/1e9:.2f}B)")
print()

total_bytes = bf16_bytes + aqlm_total_bytes
total_orig = 9750000000  # ~9.75 GB source
print(f"  ── 权重大小 ──")
print(f"  bf16 组件:        {bf16_bytes/1e9:.2f} GB")
print(f"  AQLM Attn 存储:   {attn_aqlm_bytes/1e6:.0f} MB")
print(f"  AQLM MLP 存储:    {mlp_aqlm_bytes/1e6:.0f} MB")
print(f"  总存储:           {total_bytes/1e9:.2f} GB  (vs 源 ~{total_orig/1e9:.2f} GB)")
print(f"  整体压缩比:       {total_orig/total_bytes:.1f}x  节省: {(1-total_bytes/total_orig)*100:.1f}%")
print()

print(f"  ── AQLM 每组投影结构 ──")
def print_proj(name, info, orig_shape, orig_bytes):
    cb_shape, codes_shape, scales_shape, cb_b, codes_b, scales_b, total = info
    print(f"  [{name}] 原始 {orig_shape} = {orig_bytes/1024:.0f} KB")
    print(f"    codebooks  {cb_shape}  = {cb_b/1024:.1f} KB  (bf16)")
    print(f"    codes      {codes_shape}  = {codes_b/1024:.1f} KB  ({'int16' if '14' in name else 'int8'})")
    print(f"    scales     {scales_shape}  = {scales_b/1024:.1f} KB  (bf16)")
    print(f"    单组合计   = {total/1024:.1f} KB  ({orig_bytes/total:.1f}x 压缩)")
    print()

print_proj("MLP gate_proj  (n=16, [9728,2560])", mlp_proj, "[9728,2560]", 9728*2560*2)
print_proj("Attn q_proj   (n=16, [2560,2560])", attn_q_proj, "[2560,2560]", 2560*2560*2)
print_proj("Attn k_proj   (n=14, [1024,2560])", attn_k_proj, "[1024,2560]", 1024*2560*2)

print(f"  ── 按组件权重大小 ──")
print(f"  {'组件':<35} {'大小':>10} {'占比':>7}")
print(f"  {'─'*35} {'─'*10} {'─'*7}")

for cat in sorted(cat_bytes.keys(), key=lambda c: cat_bytes[c], reverse=True):
    cb = cat_bytes[cat]
    if cb > 0:
        pct = cb / total_bytes * 100
        print(f"  {cat:<35} {cb/1e6:>8.1f} MB {pct:>5.1f}%")

print(f"  {'─'*35} {'─'*10} {'─'*7}")
print(f"  {'合计':<35} {total_bytes/1e9:>8.2f} GB")
print()

# Attn compression summary
attn_orig = 32 * (2560*2560*2 + 1024*2560*2*2 + 2560*2560*2)  # q+k+v+o, 2 bytes each
mlp_orig = 32 * (9728*2560*2*2 + 2560*9728*2)  # gate+up+down, 2 bytes each
print(f"  ── 压缩明细 ──")
print(f"  LLM Attn L4-L35 原始:   {attn_orig/1e6:.0f} MB -> AQLM: {attn_aqlm_bytes/1e6:.0f} MB ({attn_orig/attn_aqlm_bytes:.1f}x)")
print(f"  LLM MLP  L4-L35 原始:   {mlp_orig/1e6:.0f} MB -> AQLM: {mlp_aqlm_bytes/1e6:.0f} MB ({mlp_orig/mlp_aqlm_bytes:.1f}x)")
print(f"  Attn+MLP 合计:          {(attn_orig+mlp_orig)/1e6:.0f} MB -> {(attn_aqlm_bytes+mlp_aqlm_bytes)/1e6:.0f} MB ({(attn_orig+mlp_orig)/(attn_aqlm_bytes+mlp_aqlm_bytes):.1f}x)")
print()
