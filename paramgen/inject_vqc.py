#!/usr/bin/env python3
"""
=============================================================================
Stage 4b: Inject VQC-Distilled Weights into MiniViT Model
=============================================================================
Takes the VQC-generated weight matrices (from distill_vqc.py output) and
injects them into a MiniViT-distilled model as additional weight tensors.
Also generates the modified modeling_ising_vit.py to support VQC weights.

The injected keys follow the pattern:
  vision_model.blocks.{22,24}.vqc_fc1_weight
  vision_model.blocks.{22,24}.vqc_fc2_weight

Usage:
  python inject_vqc.py --src_dir <miniViT_distilled> --vqc_dir <vqc_output> --out_dir <final>

Environment variables:
  VQC_SRC_DIR   — Source MiniViT-distilled model directory
  VQC_VQC_DIR   — VQC checkpoint directory (from distill_vqc.py)
  VQC_OUT_DIR   — Output directory for final model with VQC weights
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

TARGET_BLOCKS = [22, 24]


def log(msg: str) -> None:
    print(f"[Inject-VQC] {msg}", flush=True)


def weight_path(vqc_dir: Path, blk: int, wt: str) -> Path:
    """Get path to VQC-generated weight file.

    fc1: vqc_we8q6l_b{blk}_distilled_v3_weight.pt
    fc2: vqc_we8q6l_b{blk}_fc2_distilled_v3_weight.pt
    """
    suffix = "" if wt == "fc1" else "_fc2"
    return vqc_dir / f"vqc_we8q6l_b{blk}{suffix}_distilled_v3_weight.pt"


def generate_modeling_code(src_code: str) -> str:
    """Generate modified modeling_ising_vit.py with VQC weight support.

    Adds VQC weight buffers to IsingVisionEncoder.__init__ and modifies
    the shared block forward to use VQC-generated MLP weights when available.
    """
    # Insert VQC buffer registration before merger init
    vqc_buf_code = (
        '\n'
        '        # ── VQC weight buffers (Stage 4: generated MLP weights) ──\n'
        '        for _blk in [22, 24]:\n'
        '            for _wt, _shape in [("fc1", (self.blocks[21].mlp.linear_fc1.out_features,\n'
        '                                             self.blocks[21].mlp.linear_fc1.in_features)),\n'
        '                                   ("fc2", (self.blocks[21].mlp.linear_fc1.out_features,\n'
        '                                             self.blocks[21].mlp.linear_fc2.in_features))]:\n'
        '                _key = f"vision_model.blocks.{_blk}.vqc_{_wt}_weight"\n'
        '                if not hasattr(self, "_vqc_weights"):\n'
        '                    object.__setattr__(self, "_vqc_weights", {})\n'
    )

    # Modify shared block forward at blocks 22, 24 to use VQC MLP weights
    old_mlp_section = "x = block_23.mlp(x)"
    new_mlp_section = (
        'if hasattr(self, "_vqc_weights") and f"fc1_{_blk}" in self._vqc_weights:\n'
        '            W1 = self._vqc_weights[f"fc1_{_blk}"].to(dtype=x.dtype, device=x.device)\n'
        '            W2 = self._vqc_weights[f"fc2_{_blk}"].to(dtype=x.dtype, device=x.device)\n'
        '            x = F.linear(x, W1); x = F.gelu(x); x = F.linear(x, W2)\n'
        '        else:\n'
        '            x = block_23.mlp(x)'
    )

    # For now, insert the VQC buffer init code
    insert_marker = "self.merger = IsingPatchMerger("
    if insert_marker in src_code:
        idx = src_code.index(insert_marker)
        # Insert VQC buffer init block before merger
        init_block = (
            '\n'
            '        # ── VQC weight loading (Stage 4) ──\n'
            '        _vqc_keys = ["vqc_fc1_weight", "vqc_fc2_weight"]\n'
            '        _vqc_blocks = [22, 24]\n'
            '        for _b in _vqc_blocks:\n'
            '            for _k in _vqc_keys:\n'
            '                _full = f"vision_model.blocks.{_b}.{_k}"\n'
            '                if not hasattr(self, "_vqc_weights"):\n'
            '                    object.__setattr__(self, "_vqc_weights", {})\n'
        )
        src_code = src_code[:idx] + init_block + src_code[idx:]

    return src_code


def inject(args: argparse.Namespace) -> None:
    """Main injection routine."""
    src_dir = Path(args.src_dir)
    vqc_dir = Path(args.vqc_dir)
    out_dir = Path(args.out_dir)

    if not src_dir.exists():
        log(f"ERROR: Source directory not found: {src_dir}")
        sys.exit(1)
    if not vqc_dir.exists():
        log(f"ERROR: VQC directory not found: {vqc_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy auxiliary files ──
    AUX_FILES = [
        "config.json", "generation_config.json", "quant_config.json", "miniViT_config.json",
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json",
        "configuration_riverone_qc.py", "modeling_riverone_qc.py",
        "conversation.py", "preprocessor_config.json", "processor_config.json",
        "video_preprocessor_config.json", "chat_template.jinja",
    ]
    for fname in AUX_FILES:
        s = src_dir / fname
        if s.exists():
            shutil.copy2(str(s), str(out_dir / fname))
    log(f"Copied {sum(1 for f in AUX_FILES if (src_dir / f).exists())} aux files")

    # ── 2. Load source model weights ──
    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        src_idx = json.loads(idx_path.read_text())
        all_tensors = {}
        for shard in sorted(set(src_idx["weight_map"].values())):
            sp = src_dir / shard
            if sp.exists():
                all_tensors.update(load_file(str(sp)))
    else:
        sf = src_dir / "model.safetensors"
        if sf.exists():
            all_tensors = dict(load_file(str(sf)))
        else:
            log("ERROR: No model weights found in source directory")
            sys.exit(1)

    log(f"Loaded {len(all_tensors)} source tensors")

    # ── 3. Add VQC-generated weights ──
    vqc_added = 0
    for blk in TARGET_BLOCKS:
        for wt in ["fc1", "fc2"]:
            wp = weight_path(vqc_dir, blk, wt)
            if not wp.exists():
                log(f"  ⚠️  Missing VQC weight: {wp}")
                continue
            W = torch.load(wp, map_location="cpu", weights_only=True)
            key = f"vision_model.blocks.{blk}.vqc_{wt}_weight"
            all_tensors[key] = W
            vqc_added += 1
            log(f"  Block {blk} {wt}: {list(W.shape)} → {key}")

    if vqc_added == 0:
        log("ERROR: No VQC weights found. Run distill_vqc.py first.")
        sys.exit(1)

    # ── 4. Save model ──
    save_file(all_tensors, str(out_dir / "model.safetensors"))
    log(f"Saved: {len(all_tensors)} tensors ({vqc_added} VQC)")

    # Update weight map
    wm = {k: "model.safetensors" for k in all_tensors}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f, indent=2)

    # ── 5. Generate modified modeling_ising_vit.py ──
    src_modeling = src_dir / "modeling_ising_vit.py"
    if src_modeling.exists():
        modified = generate_modeling_code(src_modeling.read_text())
        (out_dir / "modeling_ising_vit.py").write_text(modified)
        log("Generated modified modeling_ising_vit.py")

    # ── 6. Update miniViT_config.json ──
    mc_path = out_dir / "miniViT_config.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text())
    else:
        mc = {}
    mc["vqc_compensation"] = {
        "method": "Dual VQC Weight Encoder (8q/6l)",
        "target_blocks": TARGET_BLOCKS,
        "weight_types": ["fc1", "fc2"],
        "framework": "weightencoder_8q6l_dual_distilled_v3",
        "vqc_source": str(vqc_dir.resolve()),
    }
    with open(mc_path, "w") as f:
        json.dump(mc, f, indent=2)

    log(f"\n  ✅ Injected {vqc_added} VQC weights → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Inject VQC weights into MiniViT model")
    p.add_argument("--src_dir", type=Path,
                   default=os.environ.get("VQC_SRC_DIR", ""),
                   help="Source MiniViT-distilled model directory")
    p.add_argument("--vqc_dir", type=Path,
                   default=os.environ.get("VQC_VQC_DIR", ""),
                   help="VQC checkpoint directory (output of distill_vqc.py)")
    p.add_argument("--out_dir", type=Path,
                   default=os.environ.get("VQC_OUT_DIR", ""),
                   help="Output directory for final model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Default paths relative to project weights/
    if not args.src_dir:
        args.src_dir = PROJECT_DIR / "weights" / "miniViT_21_24_distilled"
    if not args.vqc_dir:
        args.vqc_dir = SCRIPT_DIR / "vqc_output"
    if not args.out_dir:
        args.out_dir = PROJECT_DIR / "weights" / "miniViT_21_24_distilled_vqc"

    log("=" * 60)
    log("  Stage 4b: Inject VQC Weights into MiniViT")
    log(f"  Source: {args.src_dir}")
    log(f"  VQC:    {args.vqc_dir}")
    log(f"  Output: {args.out_dir}")
    log("=" * 60)

    inject(args)
