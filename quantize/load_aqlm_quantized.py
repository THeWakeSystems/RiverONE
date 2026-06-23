#!/usr/bin/env python3
"""
load_aqlm_quantized.py — 加载 AQLM 量化后的 RiverOne-QC-4B-v1 模型

使用方式:
  from load_aqlm_quantized import load_quantized_model
  model, tokenizer = load_quantized_model(
      "/home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/RiverOne-QC-4B-v1-AQLM-36L"
  )

原理:
  AQLM 量化后将 nn.Linear.weight 替换为 .codes + .codebooks + .scales,
  transformers 的 from_pretrained 不认这种格式。
  本脚本手动加载这些参数并替换为 aqlm.inference.QuantizedLinear。

适用模型:
  RiverOne-QC-4B-v1（miniViT 视觉编码器 + Qwen3-4B LLM）
  量化范围: LLM 全部 36 层 Attention + MLP 线性权重
  量化方案: AQLM 1×16
"""
from __future__ import annotations

import sys
import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer
from aqlm import QuantizedLinear as AQLMLinear


def load_quantized_model(
    model_dir: str,
    dtype: str = "bfloat16",
    device: str = "auto",
):
    """加载 AQLM 量化后的 RiverOne-QC-4B-v1 模型。

    自动识别量化层（.codes/.codebooks/.scales），
    替换 nn.Linear -> QuantizedLinear。

    Args:
        model_dir: 量化模型目录路径。
        dtype: 模型精度（bfloat16/float16/float32）。
        device: 设备（auto/cuda:0/cpu）。

    Returns:
        (model, tokenizer) 元组。
    """
    model_path = Path(model_dir).resolve()
    assert model_path.exists(), (
        f"模型目录不存在: {model_dir}"
    )

    # 将模型目录加入 sys.path（用于 trust_remote_code 加载）
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # ── 1. 读取权重索引，识别量化层 ────────────────────────
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # 分组：按 codebooks 键识别量化层
    quantized_groups: Dict[str, dict] = defaultdict(dict)
    non_quantized_keys: list[str] = []

    for key, shard in weight_map.items():
        if key.endswith(".codebooks"):
            # e.g. "language_model.model.layers.32.mlp.down_proj"
            base = key[: -len(".codebooks")]
            quantized_groups[base]["codebooks_file"] = shard
            quantized_groups[base]["codebooks_key"] = key
        elif key.endswith(".codes"):
            base = key[: -len(".codes")]
            quantized_groups[base]["codes_file"] = shard
            quantized_groups[base]["codes_key"] = key
        elif key.endswith(".scales"):
            base = key[: -len(".scales")]
            quantized_groups[base]["scales_file"] = shard
            quantized_groups[base]["scales_key"] = key
        else:
            non_quantized_keys.append(key)

    # 检查量化层完整性（必须同时有 codebooks、codes、scales）
    valid_groups = {}
    for base, info in quantized_groups.items():
        if all(
            k in info
            for k in [
                "codebooks_file",
                "codes_file",
                "scales_file",
            ]
        ):
            valid_groups[base] = info

    print(
        f"[AQLM] 检测到 {len(valid_groups)} 个量化层"
    )
    for base in sorted(valid_groups.keys()):
        print(f"        {base}")

    # ── 2. 加载所有 safetensors 分片 ────────────────────────
    shard_files = sorted(
        set(
            info[k]
            for info in valid_groups.values()
            for k in [
                "codebooks_file",
                "codes_file",
                "scales_file",
            ]
        )
        | set(weight_map[k] for k in non_quantized_keys)
    )

    all_tensors: Dict[str, torch.Tensor] = {}
    for shard_name in shard_files:
        shard_path = model_path / shard_name
        if shard_path.exists():
            tensors = load_file(str(shard_path))
            all_tensors.update(tensors)

    # ── 3. 加载模型骨架（不含量化层权重）──────────────────
    print(f"[AQLM] 加载模型骨架 ({dtype}) ...")

    from modeling_riverone_qc import RiverOneQCModel

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RiverOneQCModel.from_pretrained(
            str(model_path), **load_kwargs
        )

    # 统一移到目标设备
    if device != "auto":
        model = model.to(device)
    elif torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()

    # ── 4. 替换量化层：nn.Linear -> aqlm QuantizedLinear ───
    print(
        f"[AQLM] 替换量化层为 QuantizedLinear ..."
    )
    llm_layers = model.language_model.model.layers

    for base, info in valid_groups.items():
        # 解析层索引和子层名:
        # "language_model.model.layers.32.mlp.down_proj"
        parts = base.split(".")
        # parts = ["language_model", "model", "layers",
        #          "32", "mlp", "down_proj"]
        layer_idx = int(parts[3])
        # e.g. ["mlp", "down_proj"] or
        #      ["self_attn", "q_proj"]
        sublayer_path = parts[4:]

        # 加载量化参数
        # [1, 65536, 1, 16]
        codebooks = all_tensors[info["codebooks_key"]]
        # [out_groups, in_groups, 1]
        codes = all_tensors[info["codes_key"]]
        # [out_groups, 1, 1, 1]
        scales = all_tensors[info["scales_key"]]

        (
            num_codebooks,
            codebook_size,
            out_group_size,
            in_group_size,
        ) = codebooks.shape
        num_out_groups, num_in_groups, _ = codes.shape
        in_features = num_in_groups * in_group_size
        out_features = num_out_groups * out_group_size

        # 检查是否有 bias（原始模型可能有）
        bias_key = f"{base}.bias"
        bias = all_tensors.get(bias_key, None)
        has_bias = bias is not None

        # 创建推理版 QuantizedLinear
        qlinear = AQLMLinear(
            in_features=in_features,
            out_features=out_features,
            in_group_size=in_group_size,
            out_group_size=out_group_size,
            num_codebooks=num_codebooks,
            nbits_per_codebook=codebook_size.bit_length()
            - 1,  # 65536 -> 16
            bias=has_bias,
            device=codebooks.device,
            dtype=codebooks.dtype,
        )

        # 赋值参数
        qlinear.codebooks.data.copy_(codebooks)
        qlinear.codes.data.copy_(
            codes.to(qlinear.codes.dtype)
        )
        qlinear.scales.data.copy_(scales)
        if has_bias:
            qlinear.bias.data.copy_(bias)

        # 在模型中找到对应子模块并替换
        target_layer = llm_layers[layer_idx]
        parent = target_layer
        for seg in sublayer_path[:-1]:
            parent = getattr(parent, seg)
        # 移动到与父模块相同设备
        qlinear = qlinear.to(
            device=next(parent.parameters()).device
        )
        setattr(parent, sublayer_path[-1], qlinear)

    print(
        f"[AQLM] 量化层替换完成，"
        f"共 {len(valid_groups)} 层"
    )

    # ── 5. 加载 tokenizer ──────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )

    print(
        f"[AQLM] 模型加载完成  "
        f"num_image_token={model.num_image_token}"
    )
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# CLI 测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="加载 AQLM 量化后的 RiverOne-QC-4B-v1 模型"
    )
    parser.add_argument(
        "--model-dir",
        default=(
            "/home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/"
            "RiverOne-QC-4B-v1-AQLM-36L"
        ),
        help="量化模型目录路径",
    )
    parser.add_argument(
        "--question",
        default="Hello, who are you?",
        help="测试问题",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="最大生成 token 数",
    )
    args = parser.parse_args()

    model, tokenizer = load_quantized_model(args.model_dir)

    print(f"\n{'='*60}")
    print(f"测试问题: {args.question}")
    print(f"{'='*60}")

    response, _ = model.chat(
        tokenizer=tokenizer,
        pixel_values=None,
        question=args.question,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
    )
    print(f"模型回答: {response}")
