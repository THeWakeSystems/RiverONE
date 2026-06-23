#!/usr/bin/env python3
"""Evaluate perplexity of a RiverOne-QC model on QcalEval validation set.

用途：PV-Tuning 前后对比验证，测量 QcalEval 上的 assistant-token perplexity。
同时支持原版模型与 PV-tuned 模型。

用法:
    # 评估 PV-tuning 前的模型
    python3 evaluate_perplexity.py --model_dir ../miniViT/miniViT_distilled

    # 评估 PV-tuning 后的模型
    python3 evaluate_perplexity.py --model_dir outputs/pv_tuned_qcaleval

    # 只评估零样本 split
    python3 evaluate_perplexity.py --model_dir ../miniViT/miniViT_distilled --split zs

    # 只评估上下文学习 split
    python3 evaluate_perplexity.py --model_dir ../miniViT/miniViT_distilled --split icl

    # 允许丢失图像（仅冒烟测试）
    python3 evaluate_perplexity.py --model_dir ../miniViT/miniViT_distilled --allow_missing_images
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

PV_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PV_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "weights" / "miniViT_distilled"
DEFAULT_DATA_DIR = PV_DIR / "QcalEval"

# 复用训练脚本中的图像预处理和数据集逻辑
sys.path.insert(0, str(PV_DIR))
from train_pv_tuning import (
    QcalEvalSFTDataset,
    QcalEvalCollator,
    load_model_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perplexity on QcalEval")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=["zs", "icl", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_json", type=Path, default=None, help="Save detailed results to JSON")
    return parser.parse_args()


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)

    print(f"[Eval] Loading model from {args.model_dir}")
    model, tokenizer = load_model_and_tokenizer(args)

    # 加载 PV-Tuned 的 AQLM 权重（与训练时相同流程）
    from train_pv_tuning import replace_aqlm_layers_for_training
    import torch as _torch
    master_dtype = _torch.float32
    quantized = replace_aqlm_layers_for_training(
        model, args.model_dir,
        master_dtype=master_dtype,
        buffer_dtype=_torch.bfloat16,
        use_proxy=False,
        max_layers=None,
    )
    print(f"[Eval] replaced {len(quantized)} AQLM layers with tuned weights")

    model.to(device)
    model.eval()

    split_map = {
        "zs": ["qcaleval_zs_sft.jsonl"],
        "icl": ["qcaleval_icl_sft.jsonl"],
        "all": ["qcaleval_zs_sft.jsonl", "qcaleval_icl_sft.jsonl"],
    }
    train_files = split_map[args.split]

    dataset = QcalEvalSFTDataset(
        data_dir=args.data_dir,
        train_files=train_files,
        tokenizer=tokenizer,
        num_image_token=model.num_image_token,
        max_length=args.max_length,
        image_size=448,
        min_tiles=1,
        max_tiles=12,
        use_thumbnail=True,
        allow_missing_images=args.allow_missing_images,
    )

    if args.max_samples > 0 and args.max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[: args.max_samples].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"[Eval] Subsampled {args.max_samples} examples")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=QcalEvalCollator(tokenizer, model.num_image_token),
        pin_memory=device.type == "cuda",
    )

    per_sample_losses = []
    per_sample_ppl = []
    total_tokens = 0
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Evaluating")
    for batch in progress:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(dtype=model.dtype)

        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            print(f"[Eval] WARNING: non-finite loss, skipping batch")
            continue

        # Per-sample tracking
        batch_size_actual = batch["input_ids"].shape[0]
        for i in range(batch_size_actual):
            # Compute per-sample loss
            sample_input_ids = batch["input_ids"][i:i+1]
            sample_labels = batch["labels"][i:i+1]
            sample_pixel = batch["pixel_values"][i:i+1] if batch["pixel_values"].ndim > 0 else batch["pixel_values"]
            sample_flags = batch["image_flags"][i:i+1] if "image_flags" in batch else None

            with torch.autocast(device_type=device.type, dtype=model.dtype, enabled=True):
                sample_out = model(
                    pixel_values=sample_pixel,
                    input_ids=sample_input_ids,
                    labels=sample_labels,
                    image_flags=sample_flags,
                )
            sample_loss = sample_out.loss.item() if sample_out.loss is not None else float("nan")
            sample_ppl = compute_perplexity(sample_loss)
            per_sample_losses.append(sample_loss)
            per_sample_ppl.append(sample_ppl)

        n_tokens = (batch["labels"] != -100).sum().item()
        total_tokens += n_tokens
        total_loss += loss.item() * n_tokens

        avg_ppl = compute_perplexity(total_loss / max(total_tokens, 1))
        progress.set_postfix(ppl=f"{avg_ppl:.2f}", tokens=total_tokens)

    avg_loss = total_loss / max(total_tokens, 1)
    avg_ppl = compute_perplexity(avg_loss)

    results = {
        "model_dir": str(args.model_dir),
        "split": args.split,
        "num_samples": len(per_sample_losses),
        "total_tokens": total_tokens,
        "average_loss": round(avg_loss, 6),
        "perplexity": round(avg_ppl, 4),
        "per_sample_median_ppl": round(float(torch.tensor(per_sample_ppl).nan_to_num(float("inf")).median()), 4),
        "per_sample_min_ppl": round(float(torch.tensor(per_sample_ppl).nan_to_num(float("inf")).min()), 4),
        "per_sample_max_ppl": round(float(torch.tensor(per_sample_ppl).nan_to_num(float("inf")).max()), 4),
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.model_dir}")
    print(f"  Split:          {args.split}")
    print(f"  Samples:        {results['num_samples']}")
    print(f"  Total tokens:   {results['total_tokens']}")
    print(f"  Average loss:   {results['average_loss']:.4f}")
    print(f"  Perplexity:     {results['perplexity']:.2f}")
    print(f"  Median PPL:     {results['per_sample_median_ppl']:.2f}")
    print(f"  Min PPL:        {results['per_sample_min_ppl']:.2f}")
    print(f"  Max PPL:        {results['per_sample_max_ppl']:.2f}")
    print(f"{'='*60}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Results saved to {args.output_json}")

    return results


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
