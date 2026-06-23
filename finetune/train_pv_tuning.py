#!/usr/bin/env python3
"""PV-tune RiverOne-QC-4B-v1-AQLM-miniViT on QcalEval SFT JSONL data.

This script keeps the stored AQLM representation intact:
  - P step: update continuous AQLM codebooks/scales by backprop.
  - V step: update discrete AQLM codes with the local L2 beam-search step.

The objective is supervised cross entropy on assistant tokens from QcalEval.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


PV_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PV_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "weights" / "miniViT_v2_distilled"
DEFAULT_DATA_DIR = PV_DIR / "QcalEval"
DEFAULT_OUTPUT_DIR = PV_DIR / "outputs_v2" / "pv_tuned_qcaleval"
AQLM_LIB_DIR = PROJECT_DIR / "engine"

if str(AQLM_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(AQLM_LIB_DIR))

from src.beam_search_l2 import beam_search_optimal_codes  # noqa: E402
from src.utils import _dequantize_weight  # noqa: E402


ISING_MEAN = (0.5, 0.5, 0.5)
ISING_STD = (0.5, 0.5, 0.5)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=["qcaleval_zs_sft.jsonl", "qcaleval_icl_sft.jsonl"],
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=Path, default=PV_DIR / "cache")

    parser.add_argument("--device", default="auto" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for model parallelism (1-4)")
    parser.add_argument("--load_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--master_dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--buffer_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--amp_dtype", default="bfloat16", choices=["none", "bfloat16", "float16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--min_tiles", type=int, default=1)
    parser.add_argument("--max_tiles", type=int, default=12)
    parser.add_argument("--use_thumbnail", action="store_true", default=True)
    parser.add_argument("--allow_missing_images", action="store_true")

    parser.add_argument("--update_codes", action="store_true", default=True)
    parser.add_argument("--no_update_codes", action="store_false", dest="update_codes")
    parser.add_argument("--update_codebooks_and_scales", action="store_true", default=True)
    parser.add_argument("--no_update_codebooks_and_scales", action="store_false", dest="update_codebooks_and_scales")
    parser.add_argument("--update_non_quantized_parameters", action="store_true")
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--no_freeze_vision", action="store_false", dest="freeze_vision")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--code_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.90)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--lr_scheduler", default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Override warmup_ratio with explicit step count")

    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_code_change_per_step", type=float, default=1e-3)
    parser.add_argument("--code_trust_ratio", type=float, default=None)
    parser.add_argument("--code_update_every", type=int, default=1)
    parser.add_argument("--delta_decay", type=float, default=0.0)
    parser.add_argument("--max_quantized_layers", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--log_every_steps", type=int, default=1)
    parser.add_argument("--dry_run_steps", type=int, default=0)

    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainableAQLMLinear(nn.Module):
    """Training-time AQLM Linear with differentiable dequantization."""

    def __init__(
        self,
        *,
        codebooks: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        master_dtype: torch.dtype,
        buffer_dtype: torch.dtype,
        use_proxy: bool,
    ):
        super().__init__()
        num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
        num_out_groups, num_in_groups, _ = codes.shape

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.out_group_size = out_group_size
        self.in_group_size = in_group_size
        self.out_features = num_out_groups * out_group_size
        self.in_features = num_in_groups * in_group_size

        self.codebooks = nn.Parameter(codebooks.to(master_dtype), requires_grad=True)
        self.scales = nn.Parameter(scales.to(master_dtype), requires_grad=True)
        codes_data = codes.detach().clone()
        if torch.iinfo(codes_data.dtype).bits < 32:
            codes_data = codes_data.to(torch.int32)
            # int16 负值代表高位 code（32768..65535），显式偏移修正
            codes_data = torch.where(codes_data < 0, codes_data + codebook_size, codes_data)
        self.codes = nn.Parameter(codes_data, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.to(master_dtype), requires_grad=True)

        if use_proxy:
            with torch.no_grad():
                proxy = self.dequantize(dtype=buffer_dtype)
            self.weight_proxy = nn.Parameter(proxy, requires_grad=True)
        else:
            self.register_parameter("weight_proxy", None)

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        weight = _dequantize_weight(self.codes, self.codebooks, self.scales)
        return weight if dtype is None else weight.to(dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize(dtype=input.dtype)
        if self.weight_proxy is not None:
            proxy = self.weight_proxy.to(dtype=input.dtype, device=input.device)
            weight = weight + (proxy - proxy.detach())
        bias = self.bias.to(dtype=input.dtype, device=input.device) if self.bias is not None else None
        return F.linear(input, weight, bias)

    @torch.no_grad()
    def pv_update_codes_(
        self,
        *,
        beam_size: int,
        max_update_fraction: float,
        trust_ratio: Optional[float],
        delta_decay: float,
    ) -> float:
        if self.weight_proxy is None:
            return 0.0
        prev_codes = self.codes.detach().clone()
        reference = self.weight_proxy.detach().to(device=self.codebooks.device, dtype=self.codebooks.dtype)
        new_codes = beam_search_optimal_codes(
            reference_weight=reference,
            codebooks=self.codebooks.detach(),
            prev_codes=self.codes.detach(),
            scales=self.scales.detach(),
            beam_size=beam_size,
            max_update_fraction=max_update_fraction,
            trust_ratio=trust_ratio,
        )
        self.codes.copy_(new_codes.to(self.codes.dtype))
        changed = torch.not_equal(prev_codes, self.codes).any(dim=-1).float().mean().item()

        if delta_decay > 0:
            quantized = self.dequantize(dtype=self.weight_proxy.dtype)
            self.weight_proxy.mul_(1.0 - delta_decay).add_(quantized, alpha=delta_decay)
        return changed


def _module_get(root: nn.Module, parts: list[str]) -> nn.Module:
    cur = root
    for part in parts:
        if isinstance(cur, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _module_set(root: nn.Module, parts: list[str], module: nn.Module) -> None:
    parent = _module_get(root, parts[:-1])
    leaf = parts[-1]
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _find_quantized_groups(model_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for key in weight_map:
        if key.endswith(".codebooks"):
            groups[key[: -len(".codebooks")]]["codebooks"] = key
        elif key.endswith(".codes"):
            groups[key[: -len(".codes")]]["codes"] = key
        elif key.endswith(".scales"):
            groups[key[: -len(".scales")]]["scales"] = key

    valid = {base: info for base, info in groups.items() if {"codebooks", "codes", "scales"} <= set(info)}
    return valid, weight_map


def replace_aqlm_layers_for_training(
    model: nn.Module,
    model_dir: Path,
    *,
    master_dtype: torch.dtype,
    buffer_dtype: torch.dtype,
    use_proxy: bool,
    max_layers: Optional[int],
) -> list[tuple[str, TrainableAQLMLinear]]:
    groups, weight_map = _find_quantized_groups(model_dir)
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def tensor_for(key: str) -> torch.Tensor:
        shard_name = weight_map[key]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(str(model_dir / shard_name), device="cpu")
        return shard_cache[shard_name][key]

    quantized_modules: list[tuple[str, TrainableAQLMLinear]] = []
    for base in sorted(groups):
        if max_layers is not None and len(quantized_modules) >= max_layers:
            break
        info = groups[base]
        old_module = _module_get(model, base.split("."))
        bias_key = f"{base}.bias"
        bias = tensor_for(bias_key) if bias_key in weight_map else getattr(old_module, "bias", None)
        if isinstance(bias, nn.Parameter):
            bias = bias.detach().cpu()

        new_module = TrainableAQLMLinear(
            codebooks=tensor_for(info["codebooks"]),
            codes=tensor_for(info["codes"]),
            scales=tensor_for(info["scales"]),
            bias=bias,
            master_dtype=master_dtype,
            buffer_dtype=buffer_dtype,
            use_proxy=use_proxy,
        )
        # 放置到旧模块所在 GPU（device_map="auto" 后各层可能在不同卡上）
        old_device = next(old_module.parameters()).device
        new_module.to(old_device)
        _module_set(model, base.split("."), new_module)
        quantized_modules.append((base, new_module))

    if not quantized_modules:
        raise RuntimeError(f"No AQLM quantized layers found in {model_dir}")
    return quantized_modules


def _build_transform(image_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=ISING_MEAN, std=ISING_STD),
        ]
    )


def _find_best_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Iterable[tuple[int, int]],
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = image_size**2
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_ratio_diff or (
            diff == best_ratio_diff and ratio[0] * ratio[1] > best_ratio[0] * best_ratio[1]
        ):
            best_ratio_diff = diff
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    n_cols, n_rows = _find_best_aspect_ratio(aspect_ratio, target_ratios, image_size)
    resized = image.resize((image_size * n_cols, image_size * n_rows), Image.BICUBIC)
    transform = _build_transform(image_size)

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
            tiles.append(transform(resized.crop(box)))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(transform(image.resize((image_size, image_size), Image.BICUBIC)))
    return torch.stack(tiles, dim=0)


class QcalEvalSFTDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: Path,
        train_files: list[str],
        tokenizer,
        num_image_token: int,
        max_length: int,
        image_size: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        allow_missing_images: bool,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.max_length = max_length
        self.image_size = image_size
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.allow_missing_images = allow_missing_images
        self.records = []

        for filename in train_files:
            path = data_dir / filename
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        record["_source_file"] = filename
                        self.records.append(record)
        if not self.records:
            raise RuntimeError(f"No samples loaded from {data_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, image_path: str) -> Path:
        path = Path(image_path)
        return path if path.is_absolute() else self.data_dir / path

    def _load_images(self, image_field) -> tuple[torch.Tensor, list[int]]:
        paths = image_field if isinstance(image_field, list) else [image_field]
        all_tiles = []
        num_patches = []
        for rel_path in paths:
            full_path = self._resolve_image(rel_path)
            if full_path.exists():
                image = Image.open(full_path).convert("RGB")
            elif self.allow_missing_images:
                image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
            else:
                raise FileNotFoundError(
                    f"Missing image: {full_path}. Put QcalEval images under {self.data_dir}/images "
                    "or rerun with --allow_missing_images for a plumbing test."
                )
            tiles = dynamic_preprocess(
                image,
                min_num=self.min_tiles,
                max_num=self.max_tiles,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
            all_tiles.append(tiles)
            num_patches.append(tiles.shape[0])
        return torch.cat(all_tiles, dim=0), num_patches

    def _expand_image_tokens(self, text: str, num_patches: list[int]) -> str:
        expanded = text
        placeholders = expanded.count("<image>")
        if placeholders > len(num_patches):
            # 占位符多于图片：用最后一组 tile 填充多余占位符
            if len(num_patches) > 0:
                num_patches = num_patches + [num_patches[-1]] * (placeholders - len(num_patches))
            else:
                return expanded  # 无图片，原样返回
        if placeholders < len(num_patches):
            expanded = "\n".join(["<image>"] * (len(num_patches) - placeholders)) + "\n" + expanded

        for n_tiles in num_patches:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.num_image_token * n_tiles) + IMG_END_TOKEN
            expanded = expanded.replace("<image>", image_tokens, 1)
        return expanded

    def _tokenize_pair(self, question: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
        user_msg = {"role": "user", "content": question}
        assistant_msg = {"role": "assistant", "content": answer}
        prefix = self.tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
        full = self.tokenizer.apply_chat_template([user_msg, assistant_msg], tokenize=False, add_generation_prompt=False)

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        input_ids = full_ids[: self.max_length]
        labels = [-100] * min(len(prefix_ids), len(input_ids))
        labels.extend(input_ids[len(labels) :])
        labels = labels[: self.max_length]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        conversations = record["conversations"]
        if len(conversations) != 2 or conversations[0]["role"] != "user" or conversations[1]["role"] != "assistant":
            raise ValueError(f"Expected one user/assistant turn at sample index {idx}")

        pixel_values, num_patches = self._load_images(record["image"])
        question = self._expand_image_tokens(conversations[0]["content"], num_patches)
        input_ids, labels = self._tokenize_pair(question, conversations[1]["content"])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixel_values,
        }


class QcalEvalCollator:
    def __init__(self, tokenizer, num_image_token: int):
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(example["input_ids"].numel() for example in examples)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        pixel_values = []

        for example in examples:
            length = example["input_ids"].numel()
            pad = max_len - length
            input_ids.append(F.pad(example["input_ids"], (0, pad), value=pad_id))
            labels.append(F.pad(example["labels"], (0, pad), value=-100))
            attention_mask.append(F.pad(example["attention_mask"], (0, pad), value=0))
            pixel_values.append(example["pixel_values"])

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "pixel_values": torch.cat(pixel_values, dim=0),
        }
        expected_image_tokens = batch["pixel_values"].shape[0] * self.num_image_token
        actual_image_tokens = (batch["input_ids"] == self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)).sum().item()
        if expected_image_tokens != actual_image_tokens:
            if actual_image_tokens < expected_image_tokens:
                # max_length 截断了尾部 IMG_CONTEXT token，同步裁剪 pixel_values
                keep_tiles = actual_image_tokens // self.num_image_token
                if keep_tiles > 0:
                    batch["pixel_values"] = batch["pixel_values"][:keep_tiles]
                    expected_image_tokens = keep_tiles * self.num_image_token
                else:
                    batch["pixel_values"] = batch["pixel_values"][:0]
                    expected_image_tokens = 0
                # 清理 input_ids 中超出 pixel_values 的 IMG_CONTEXT token
                if actual_image_tokens > expected_image_tokens:
                    excess = actual_image_tokens - expected_image_tokens
                    img_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                    mask = (batch["input_ids"] == img_id)
                    # 从尾部开始把多余的 IMG_CONTEXT 替换为 pad
                    excess_indices = mask.nonzero(as_tuple=False)[-excess:]
                    for idx in excess_indices:
                        batch["input_ids"][idx[0], idx[1]] = pad_id
            else:
                # actual > expected: truncation removed pixel_values tiles but left image tokens
                keep_tiles = expected_image_tokens // self.num_image_token
                if keep_tiles > 0:
                    batch["pixel_values"] = batch["pixel_values"][:keep_tiles]
                else:
                    batch["pixel_values"] = batch["pixel_values"][:0]
                # Replace excess IMG_CONTEXT tokens with pad
                excess = actual_image_tokens - expected_image_tokens
                img_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                mask = (batch["input_ids"] == img_id)
                excess_indices = mask.nonzero(as_tuple=False)[-excess:]
                for idx in excess_indices:
                    batch["input_ids"][idx[0], idx[1]] = pad_id
        batch["image_flags"] = torch.ones(batch["pixel_values"].shape[0], 1, dtype=torch.long)
        return batch


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = torch_dtype(args.load_dtype)
    # Always load on CPU first to avoid OOM from random-initialized quantized layers
    # then move to GPU after AQLM layer replacement
    kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    model = AutoModel.from_pretrained(str(args.model_dir), **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def configure_training(
    model: nn.Module,
    quantized_modules: list[tuple[str, TrainableAQLMLinear]],
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    for param in model.parameters():
        param.requires_grad = False

    quant_param_ids = set()
    continuous_params = []
    proxy_params = []
    for _, module in quantized_modules:
        for param in (module.codebooks, module.scales):
            param.requires_grad = args.update_codebooks_and_scales
            quant_param_ids.add(id(param))
            if args.update_codebooks_and_scales:
                continuous_params.append(param)
        module.codes.requires_grad = False
        quant_param_ids.add(id(module.codes))
        if module.weight_proxy is not None:
            module.weight_proxy.requires_grad = args.update_codes
            quant_param_ids.add(id(module.weight_proxy))
            if args.update_codes:
                proxy_params.append(module.weight_proxy)
        if module.bias is not None:
            module.bias.requires_grad = args.update_non_quantized_parameters

    non_quantized_params = []
    if args.update_non_quantized_parameters:
        for name, param in model.named_parameters():
            if id(param) in quant_param_ids:
                continue
            if not torch.is_floating_point(param):
                continue
            if args.freeze_vision and name.startswith("vision_model."):
                continue
            param.requires_grad = True
            non_quantized_params.append(param)

    param_groups = []
    betas = (args.adam_beta1, args.adam_beta2)
    if continuous_params:
        param_groups.append({"params": continuous_params, "lr": args.lr, "betas": betas, "weight_decay": args.weight_decay})
    if non_quantized_params:
        param_groups.append({"params": non_quantized_params, "lr": args.lr, "betas": betas, "weight_decay": args.weight_decay})
    if proxy_params:
        param_groups.append({"params": proxy_params, "lr": args.code_lr, "betas": (0.0, args.adam_beta2), "weight_decay": 0.0})
    if not param_groups:
        raise RuntimeError("No trainable parameters selected")

    print(
        "[PV] trainable params: "
        f"continuous={sum(p.numel() for p in continuous_params):,}, "
        f"proxy={sum(p.numel() for p in proxy_params):,}, "
        f"non_quantized={sum(p.numel() for p in non_quantized_params):,}"
    )
    return torch.optim.AdamW(param_groups)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create LR scheduler with optional warmup."""
    if args.lr_scheduler == "none":
        return None

    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    else:
        return None

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return scheduler.get_last_lr()[0] / args.lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def pv_update_all_codes(
    quantized_modules: list[tuple[str, TrainableAQLMLinear]],
    args: argparse.Namespace,
) -> dict[str, float]:
    changes = {}
    for name, module in quantized_modules:
        changed = module.pv_update_codes_(
            beam_size=args.beam_size,
            max_update_fraction=args.max_code_change_per_step,
            trust_ratio=args.code_trust_ratio,
            delta_decay=args.delta_decay,
        )
        changes[name] = changed
    return changes


def copy_model_side_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    skip_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    skip_names = {"model.safetensors.index.json"}
    for item in source.iterdir():
        if item.name in skip_names or item.suffix in skip_suffixes:
            continue
        target = destination / item.name
        if item.is_dir():
            if item.name == "__pycache__":
                continue
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def filtered_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    result = {}
    for name, tensor in state.items():
        if name.endswith(".weight_proxy"):
            continue
        # Fix: DeviceAwareWrapper adds ".module." prefix to layers 18-35 keys.
        # Strip it so saved checkpoint matches original model architecture.
        name = name.replace(".module.", ".")
        result[name] = tensor
    return result


def save_model(model: nn.Module, tokenizer, args: argparse.Namespace, step: Optional[int] = None) -> Path:
    output_dir = args.output_dir if step is None else args.output_dir / f"checkpoint-step-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_model_side_files(args.model_dir, output_dir)
    model.save_pretrained(
        str(output_dir),
        state_dict=filtered_state_dict(model),
        safe_serialization=False,  # MiniViT 有共享权重层(23/24)，不能用 safetensors
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def main() -> None:
    args = parse_args()
    if not 0 <= args.delta_decay <= 1:
        raise ValueError("--delta_decay must be in [0, 1]")
    if not 0 < args.max_code_change_per_step <= 1:
        raise ValueError("--max_code_change_per_step must be in (0, 1]")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    master_dtype = torch_dtype(args.master_dtype)
    buffer_dtype = torch_dtype(args.buffer_dtype)
    quantized_modules = replace_aqlm_layers_for_training(
        model,
        args.model_dir,
        master_dtype=master_dtype,
        buffer_dtype=buffer_dtype,
        use_proxy=args.update_codes,
        max_layers=args.max_quantized_layers,
    )
    print(f"[PV] replaced {len(quantized_modules)} AQLM layers with training modules")

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Second GC pass: ensures old nn.Linear random weights from replaced layers
    # are freed before model.to(device) copies everything to GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Model was loaded on CPU; move to target device(s) after layer replacement
    num_layers = len(model.language_model.model.layers)
    if args.num_gpus >= 2 and torch.cuda.device_count() >= 2:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        layers_per_gpu = num_layers // num_gpus
        splits = []  # [(start_layer, end_layer, gpu_id), ...]
        for g in range(num_gpus):
            start = g * layers_per_gpu
            # Last GPU takes all remaining layers
            end = num_layers if g == num_gpus - 1 else (g + 1) * layers_per_gpu
            splits.append((start, end, g))
        split_desc = ", ".join(f"layers {s}-{e-1}→GPU{g}" for s, e, g in splits)
        print(f"[PV] splitting model across {num_gpus} GPUs ({split_desc})")
        model.to("cuda:0")  # Move everything to GPU0 first

        class DeviceAwareWrapper(nn.Module):
            """Wrapper that moves all tensor args/kWargs (including nested tuples) to module's device."""
            def __init__(self, module: nn.Module):
                super().__init__()
                self.module = module

            @staticmethod
            def _move_to(obj, device):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device) if obj.device != device else obj
                elif isinstance(obj, (tuple, list)):
                    return type(obj)(DeviceAwareWrapper._move_to(x, device) for x in obj)
                elif isinstance(obj, dict):
                    return {k: DeviceAwareWrapper._move_to(v, device) for k, v in obj.items()}
                return obj

            def forward(self, *args, **kwargs):
                device = next(self.module.parameters()).device
                args = tuple(self._move_to(a, device) for a in args)
                kwargs = {k: self._move_to(v, device) for k, v in kwargs.items()}
                return self.module(*args, **kwargs)

        layers = model.language_model.model.layers
        # Move layers to their assigned GPUs and wrap non-GPU0 blocks
        for start, end, gpu_id in splits:
            for i in range(start, end):
                if i >= end - 1 and gpu_id == num_gpus - 1:
                    # Last layer on last GPU: move output back to GPU0 for final norm/lm_head
                    class LastLayerWrapper(DeviceAwareWrapper):
                        def forward(self, *args, **kwargs):
                            output = super().forward(*args, **kwargs)
                            if isinstance(output, torch.Tensor):
                                return output.to("cuda:0")
                            elif isinstance(output, (tuple, list)):
                                return type(output)(
                                    (x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
                                    for x in output
                                )
                            return output
                    layers[i] = LastLayerWrapper(layers[i])
                elif gpu_id > 0 and i == start:
                    # First layer of each non-GPU0 block: wrap to move inputs
                    layers[i] = DeviceAwareWrapper(layers[i])
                elif gpu_id > 0 and i > start:
                    # Subsequent layers on same GPU: also wrap for consistency
                    layers[i] = DeviceAwareWrapper(layers[i])
                # Move to target GPU
                if gpu_id > 0:
                    layers[i].to(f"cuda:{gpu_id}")
        device = torch.device("cuda:0")  # batch input and embedding on GPU0
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        model.to(device)
    model.train()
    optimizer = configure_training(model, quantized_modules, args)

    dataset = QcalEvalSFTDataset(
        data_dir=args.data_dir,
        train_files=args.train_files,
        tokenizer=tokenizer,
        num_image_token=model.num_image_token,
        max_length=args.max_length,
        image_size=args.image_size,
        min_tiles=args.min_tiles,
        max_tiles=args.max_tiles,
        use_thumbnail=args.use_thumbnail,
        allow_missing_images=args.allow_missing_images,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=QcalEvalCollator(tokenizer, model.num_image_token),
        pin_memory=device.type == "cuda",
    )

    # Estimate total optimizer steps for scheduler
    steps_per_epoch = len(dataloader) // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = create_scheduler(optimizer, total_steps, args)
    if scheduler is not None:
        print(f"[PV] scheduler: {args.lr_scheduler}, warmup_steps={scheduler.warmup_steps if hasattr(scheduler, 'warmup_steps') else int(total_steps * args.warmup_ratio)}, total_steps={total_steps}")

    amp_dtype = None if args.amp_dtype == "none" else torch_dtype(args.amp_dtype)
    use_amp = device.type == "cuda" and amp_dtype is not None
    global_step = 0
    running_loss = 0.0
    running_count = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(dtype=amp_dtype or torch_dtype(args.load_dtype))
            # 截断导致 pixel_values 为空的样本直接跳过
            if batch["pixel_values"].numel() == 0:
                continue

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isfinite(loss):
                print(f"[PV] WARNING: NaN loss at step {global_step}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            running_count += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if args.update_codes and global_step % args.code_update_every == 0:
                    changes = pv_update_all_codes(quantized_modules, args)
                    mean_change = sum(changes.values()) / max(len(changes), 1)
                else:
                    mean_change = 0.0
                optimizer.zero_grad(set_to_none=True)

                if args.log_every_steps and global_step % args.log_every_steps == 0:
                    avg_loss = running_loss / max(running_count, 1)
                    elapsed = time.time() - start_time
                    print(
                        f"[PV] step={global_step} epoch={epoch + 1} "
                        f"loss={avg_loss:.6f} mean_code_change={mean_change:.8f} elapsed={elapsed:.1f}s"
                    )
                    running_loss = 0.0
                    running_count = 0

                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    saved = save_model(model, tokenizer, args, step=global_step)
                    print(f"[PV] saved checkpoint to {saved}")

                if args.dry_run_steps and global_step >= args.dry_run_steps:
                    saved = save_model(model, tokenizer, args, step=global_step)
                    print(f"[PV] dry run complete; saved checkpoint to {saved}")
                    return

    saved = save_model(model, tokenizer, args)
    print(f"[PV] training complete; saved tuned model to {saved}")


if __name__ == "__main__":
    main()
