#!/usr/bin/env python3
"""Quick test of full PV-tuning pipeline on AQLM-32L-AttnMix_miniViT"""
import torch, sys, gc, json
from collections import defaultdict
from safetensors.torch import load_file
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from src.utils import _dequantize_weight
from transformers import AutoModel

torch.backends.cuda.matmul.allow_tf32 = True

model_dir = Path("/home/lxy/AQLM-32L-AttnMix_miniViT")
print("Loading model on CPU...")
model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
gc.collect()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

with (model_dir / "model.safetensors.index.json").open() as f:
    weight_map = json.load(f)["weight_map"]
groups = defaultdict(dict)
for key in weight_map:
    if key.endswith(".codebooks"): groups[key[:-len(".codebooks")]]["codebooks"] = key
    elif key.endswith(".codes"): groups[key[:-len(".codes")]]["codes"] = key
    elif key.endswith(".scales"): groups[key[:-len(".scales")]]["scales"] = key
valid = {b: i for b, i in groups.items() if {"codebooks","codes","scales"} <= set(i)}
print(f"Found {len(valid)} quantized groups, using first 4")

def _module_get(root, parts):
    cur = root
    for part in parts:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur

def _module_set(root, parts, module):
    parent = _module_get(root, parts[:-1])
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)

shard = load_file(str(model_dir / "model.safetensors"), device="cpu")

class TrainableAQLMLinear(torch.nn.Module):
    def __init__(self, codebooks, codes, scales, bias, master_dtype, buffer_dtype, use_proxy):
        super().__init__()
        nc, cbs, ogs, igs = codebooks.shape
        nog, nig, _ = codes.shape
        self.num_codebooks=nc; self.codebook_size=cbs; self.out_group_size=ogs; self.in_group_size=igs
        self.out_features=nog*ogs; self.in_features=nig*igs
        self.codebooks=torch.nn.Parameter(codebooks.to(master_dtype), requires_grad=True)
        self.scales=torch.nn.Parameter(scales.to(master_dtype), requires_grad=True)
        # Convert int16 codes to int32 with offset fix at creation time
        codes_data = codes.detach().clone()
        if torch.iinfo(codes_data.dtype).bits < 32:
            codes_data = codes_data.to(torch.int32)
            codes_data = torch.where(codes_data < 0, codes_data + cbs, codes_data)
        self.codes=torch.nn.Parameter(codes_data, requires_grad=False)
        if bias is not None:
            self.bias=torch.nn.Parameter(bias.to(master_dtype), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        if use_proxy:
            with torch.no_grad():
                proxy = self._dequantize_raw(dtype=buffer_dtype)
            self.weight_proxy = torch.nn.Parameter(proxy, requires_grad=True)
        else:
            self.register_parameter("weight_proxy", None)

    def _dequantize_raw(self, dtype=None):
        weight = _dequantize_weight(self.codes, self.codebooks, self.scales)
        return weight if dtype is None else weight.to(dtype)

    def forward(self, input):
        weight = self._dequantize_raw(dtype=input.dtype)
        if self.weight_proxy is not None:
            proxy = self.weight_proxy.to(dtype=input.dtype, device=input.device)
            weight = weight + (proxy - proxy.detach())
        bias = self.bias.to(dtype=input.dtype, device=input.device) if self.bias is not None else None
        return torch.nn.functional.linear(input, weight, bias)

replaced = []
for base in sorted(valid)[:4]:
    info = valid[base]
    parts = base.split(".")
    bias_key = f"{base}.bias"
    bias = shard[bias_key] if bias_key in shard else None
    new_mod = TrainableAQLMLinear(
        codebooks=shard[info["codebooks"]],
        codes=shard[info["codes"]],
        scales=shard[info["scales"]],
        bias=bias,
        master_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
        use_proxy=True,
    )
    _module_set(model, parts, new_mod)
    replaced.append(base)
    print(f"  Replaced: {base}")

gc.collect()
print(f"Replaced {len(replaced)} layers. Moving to GPU...")
model.to("cuda:0")
model.train()

print("Testing forward+backward...")
input_ids = torch.randint(0, 1000, (1, 64), device="cuda")
mask = torch.ones(1, 64, device="cuda")
labels = torch.randint(0, 1000, (1, 64), device="cuda")
labels[:, :10] = -100
pixels = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device="cuda")

out = model(input_ids=input_ids, attention_mask=mask, pixel_values=pixels, labels=labels)
loss = out.loss
print(f"Loss: {loss.item():.4f}")

loss.backward()
print("Backward OK!")
print(f"GPU mem: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated")
print("SUCCESS - full pipeline works!")
