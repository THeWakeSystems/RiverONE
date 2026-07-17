#!/usr/bin/env python3
"""
=============================================================================
Stage 4 — Phase 1a: VQC WeightEncoder fc1 Training
=============================================================================
Trains VQCWeightGenerator to reconstruct teacher fc1 weights for MiniViT
shared blocks (22, 24). Uses activation loss + weight MSE with lambda schedule.

Loss: L = L_act + λ(t) · L_weight
  L_act  = MSE(F.linear(random_tokens, VQC_weight), teacher_output)
  L_weight = MSE(VQC_weight, teacher_weight)
  λ(t): 0.01 → 5.0 over 2/3 of training

Output:
  vqc_we8q6l_b{blk}_state.pt   — full VQC+Encoder+Hyper state
  vqc_we8q6l_b{blk}_weight.pt  — generated fc1 weight matrix

Usage:
  python train_vqc_fc1.py [--teacher_dir PATH] [--output_dir PATH]
=============================================================================
"""
from __future__ import annotations

import argparse, math, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vqc_models import VQCWeightGenerator

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGET_BLOCKS = [22, 24]
N_WIRES = 8; N_QLAYERS = 6; N_BLOCKS = 8
RANK = 32; LATENT = 128; PD = 48
ENC_HIDDEN = 512; STAT_DIM = 128; REUPLOAD_EVERY = 2
LAMBDA_MIN = 0.01; LAMBDA_MAX = 5.0
N_STEPS = 2000; BATCH = 512; LOG_EVERY = max(N_STEPS // 10, 100)


def log(msg: str) -> None:
    print(f"[P1-fc1] {msg}", flush=True)


def train_block(blk: int, teacher_dir: Path, out_dir: Path) -> dict:
    log("=" * 60)
    log(f"  Block {blk}: loading teacher fc1 weight...")

    teacher_sf = teacher_dir / "model.safetensors"
    with safe_open(str(teacher_sf), framework="pt", device="cpu") as f:
        fc1_w = f.get_tensor(f"vision_model.blocks.{blk}.mlp.linear_fc1.weight")
        fc1_b = f.get_tensor(f"vision_model.blocks.{blk}.mlp.linear_fc1.bias")

    fc1_w = fc1_w.to(device=DEVICE, dtype=torch.float32)
    fc1_b = fc1_b.to(device=DEVICE, dtype=torch.float32)
    out_dim, in_dim = fc1_w.shape
    log(f"  fc1.weight: [{out_dim}, {in_dim}] ({fc1_w.numel():,})")

    # ── Build model ──
    vqc_gen = VQCWeightGenerator(
        N_WIRES, N_QLAYERS, (out_dim, in_dim),
        RANK, LATENT, PD, REUPLOAD_EVERY, N_BLOCKS, STAT_DIM, ENC_HIDDEN,
    ).to(DEVICE)

    n_total = sum(p.numel() for p in vqc_gen.parameters())
    compression = fc1_w.numel() / n_total
    log(f"  Params: {n_total:,}  Compression: {compression:.1f}×")

    # ── Training data: random tokens × teacher weight ──
    NT = BATCH * 10
    xt = torch.randn(NT, in_dim, device=DEVICE, dtype=torch.float32) * 0.7
    with torch.no_grad():
        yt = F.linear(xt, fc1_w, fc1_b)
    log(f"  Training tokens: {NT:,}")

    # ── Optimizer + schedulers ──
    opt = torch.optim.AdamW(vqc_gen.parameters(), lr=2e-3, weight_decay=1e-5)
    wu = N_STEPS // 10

    def lr_fn(step): return 2e-3 * (step+1)/wu if step < wu else 2e-3 * 0.5 * (1 + math.cos(math.pi * (step-wu)/(N_STEPS-wu)))
    def lam_fn(step): return LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * min(1.0, step / (N_STEPS * 2 // 3))

    t0 = time.time(); best_cos = -1.0; best_state = None
    for s in range(N_STEPS):
        idx = torch.randint(0, NT, (BATCH,), device=DEVICE)
        xb, yb = xt[idx], yt[idx]

        for g in opt.param_groups: g["lr"] = lr_fn(s)
        opt.zero_grad()

        W_pred = vqc_gen(fc1_w).to(dtype=xb.dtype)
        lam = lam_fn(s)
        loss_act = F.mse_loss(F.linear(xb, W_pred, fc1_b), yb)
        loss_w = F.mse_loss(W_pred, fc1_w)
        loss = loss_act + lam * loss_w
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqc_gen.parameters(), 10.0)
        opt.step()

        if s % LOG_EVERY == 0 or s == N_STEPS - 1:
            with torch.no_grad():
                Wc = vqc_gen(fc1_w)
                cos = F.cosine_similarity(Wc.flatten(), fc1_w.flatten(), dim=0).item()
                if cos > best_cos:
                    best_cos = cos
                    best_state = {k: v.cpu().clone() for k, v in vqc_gen.state_dict().items()}
            elapsed = time.time() - t0
            log(f"    [{elapsed:5.0f}s] {s+1:4d}/{N_STEPS} loss={loss.item():.6f} cos={cos:.4f} λ={lam:.2f}")

    dt = time.time() - t0
    if best_state is not None:
        vqc_gen.load_state_dict(best_state)

    with torch.no_grad():
        W_final = vqc_gen(fc1_w)
        cos_final = F.cosine_similarity(W_final.flatten(), fc1_w.flatten(), dim=0).item()

    log(f"  Done: cos={cos_final:.4f} time={dt:.0f}s")

    # ── Save ──
    tag = f"we8q6l_b{blk}"
    torch.save({
        "state": vqc_gen.state_dict(), "cos": cos_final, "best_cos": best_cos,
        "block": blk, "n_wires": N_WIRES, "n_qlayers": N_QLAYERS,
        "framework": "weightencoder_8q6l", "teacher_fc1_w": fc1_w.cpu(), "teacher_fc1_b": fc1_b.cpu(),
    }, out_dir / f"vqc_{tag}_state.pt")
    torch.save(W_final.cpu(), out_dir / f"vqc_{tag}_weight.pt")
    return {"block": blk, "cos": cos_final, "time": dt}


def main():
    p = argparse.ArgumentParser(description="Phase 1a: VQC fc1 training")
    p.add_argument("--teacher_dir", type=Path, default=os.environ.get("VQC_TEACHER_DIR", str(PROJECT_DIR / "weights" / "RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP")))
    p.add_argument("--output_dir", type=Path, default=os.environ.get("VQC_OUTPUT_DIR", str(SCRIPT_DIR / "vqc_output")))
    p.add_argument("--blocks", type=int, nargs="+", default=TARGET_BLOCKS)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Phase 1a: VQC fc1 training | teacher={args.teacher_dir} | output={args.output_dir}")

    for blk in args.blocks:
        train_block(blk, args.teacher_dir, args.output_dir)

    log(f"\nDone. Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
