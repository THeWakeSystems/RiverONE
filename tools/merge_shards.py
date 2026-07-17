#!/usr/bin/env python3
"""Merge sharded safetensors into single file for VQC scripts."""
import json, sys
from safetensors import safe_open
from safetensors.torch import save_file

src_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/lxy/workspace/RiverOne/weights/pv_tuned_step2000'

idx = json.load(open(f'{src_dir}/model.safetensors.index.json'))
shards = sorted(set(idx['weight_map'].values()))

all_tensors = {}
for shard in shards:
    with safe_open(f'{src_dir}/{shard}', framework='pt', device='cpu') as f:
        for k in f.keys():
            all_tensors[k] = f.get_tensor(k)

print(f"Merged {len(all_tensors)} keys from {len(shards)} shards")
total_bytes = sum(t.numel() * t.element_size() for t in all_tensors.values())
save_file(all_tensors, f'{src_dir}/model.safetensors')
print(f"Saved merged model.safetensors ({total_bytes / 1e9:.2f} GB)")
