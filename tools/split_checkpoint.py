#!/usr/bin/env python3
"""Split a single safetensors checkpoint into sharded format with index."""
import os, json, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

src = sys.argv[1] if len(sys.argv) > 1 else '/home/lxy/workspace/RiverOne/finetune/outputs/pv_tuned_full_19k_3ep/checkpoint-step-2000'
dst = sys.argv[2] if len(sys.argv) > 2 else '/home/lxy/workspace/RiverOne/weights/pv_tuned_step2000'

Path(dst).mkdir(parents=True, exist_ok=True)

# Copy all non-safetensors files
for f in os.listdir(src):
    if f == 'model.safetensors':
        continue
    src_f = os.path.join(src, f)
    dst_f = os.path.join(dst, f)
    if os.path.isfile(src_f):
        shutil.copy2(src_f, dst_f)

print(f"Loading safetensors from {src}...")
all_tensors = {}
with safe_open(os.path.join(src, 'model.safetensors'), framework='pt', device='cpu') as f:
    for key in f.keys():
        all_tensors[key] = f.get_tensor(key)
print(f"Loaded {len(all_tensors)} keys")

# Split into 4GB shards
MAX_SHARD = 4 * 1024 * 1024 * 1024
shard = {}
shard_idx = 0
current_size = 0
weight_map = {}

for key, tensor in all_tensors.items():
    tensor = tensor.contiguous()
    ts = tensor.numel() * tensor.element_size()
    if current_size + ts > MAX_SHARD and shard:
        fname = f"model-{shard_idx+1:05d}-of-00000.safetensors"
        save_file(shard, os.path.join(dst, fname))
        for k in shard:
            weight_map[k] = fname
        print(f"  Saved shard {shard_idx+1}: {len(shard)} keys, {current_size/1e9:.2f}GB")
        shard = {}
        shard_idx += 1
        current_size = 0
    shard[key] = tensor
    current_size += ts

if shard:
    fname = f"model-{shard_idx+1:05d}-of-00000.safetensors"
    save_file(shard, os.path.join(dst, fname))
    for k in shard:
        weight_map[k] = fname
    print(f"  Saved shard {shard_idx+1}: {len(shard)} keys, {current_size/1e9:.2f}GB")
    shard_idx += 1

# Rename shards with proper total count
total_shards = shard_idx
for i in range(total_shards):
    old = os.path.join(dst, f"model-{i+1:05d}-of-00000.safetensors")
    new = os.path.join(dst, f"model-{i+1:05d}-of-{total_shards:05d}.safetensors")
    os.rename(old, new)
    for k, v in list(weight_map.items()):
        if v == f"model-{i+1:05d}-of-00000.safetensors":
            weight_map[k] = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"

# Write index
with open(os.path.join(dst, 'model.safetensors.index.json'), 'w') as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

print(f"Done: {total_shards} shards, {len(weight_map)} keys")
print(f"Output: {dst}")
