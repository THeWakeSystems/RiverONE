#!/usr/bin/env python3
"""
=============================================================================
Stage 4 — Phase 2: Dual VQC Full-Forward Distillation
=============================================================================
Loads Phase 1 VQC checkpoints, freezes encoders and student model,
trains VQC angles + Hyper only via full ViT forward distillation.

Noise images → complete student ViT forward (blocks 22/24 with VQC MLP)
→ MSE(student_ViT_output, teacher_ViT_output) + λ * MSE(VQC_weight, teacher_weight)

Prerequisites:
  Phase 1a: train_vqc_fc1.py (produces vqc_we8q6l_b{blk}_state.pt)
  Phase 1b: train_vqc_fc2.py (produces vqc_we8q6l_b{blk}_fc2_state.pt)

Output:
  vqc_we8q6l_b{blk}_distilled_v3_state.pt  — distilled state
  vqc_we8q6l_b{blk}_distilled_v3_weight.pt — final VQC-generated weight
  vqc_we8q6l_b{blk}_fc2_distilled_v3_state.pt
  vqc_we8q6l_b{blk}_fc2_distilled_v3_weight.pt

Usage:
  python distill_vqc_dual.py --teacher_dir PATH --student_dir PATH --output_dir PATH
=============================================================================
"""
from __future__ import annotations

import argparse, math, os, sys, time, gc
from pathlib import Path

import torch
import torch.nn as nn
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
SOURCE_BLOCKS = {22: 21, 24: 23}

N_WIRES = 8; N_QLAYERS = 6; N_BLOCKS = 8
RANK = 32; LATENT = 128; PD = 48; PD_FC2 = 24
ENC_HIDDEN = 512; STAT_DIM = 128; REUPLOAD_EVERY = 2

N_IMAGES = 500; IMAGE_SIZE = 448
N_STEPS = 500; BATCH_SIZE = 2; LOG_EVERY = max(N_STEPS // 10, 25)
LR_VQC_HYPER = 1e-4; WEIGHT_DECAY = 1e-5; LAMBDA_RECON = 0.1
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[P2-Dual] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
#  VQC-MLP block forward
# ═══════════════════════════════════════════════════════════════

def _vqc_mlp_block(x, block_tgt, block_src, vqc_gen_fc1, vqc_gen_fc2, t_fc1, t_fc2):
    """MiniViT shared block with VQC-generated MLP weights."""
    residual = x; x = block_tgt.norm1(x)
    B, N, C = x.shape; M = block_src.attn.num_heads; H_dim = block_src.attn.head_dim

    qkv = block_src.attn.qkv(x).reshape(B, N, 3, M, H_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H_dim)

    if hasattr(block_tgt, "attn_transform_F2_weight"):
        scores = torch.einsum("bmnl,mk->bknl", scores, block_tgt.attn_transform_F2_weight)
    attn_w = F.softmax(scores, dim=-1)
    ap = torch.einsum("bmnl,bklh->bmknh", attn_w, v)
    if hasattr(block_tgt, "attn_transform_F1_weight"):
        ap = torch.einsum("bmknh,km->bmknh", ap, block_tgt.attn_transform_F1_weight); attn_out = ap.sum(dim=2)
    else:
        attn_out = torch.einsum("bmnl,bmlh->bmnh", attn_w, v)
    attn_out = attn_out.transpose(1, 2).reshape(B, N, C); attn_out = block_src.attn.proj(attn_out)
    x = residual + attn_out

    residual = x; x = block_tgt.norm2(x)
    if hasattr(block_tgt, "mlp_dwconv"): x_t = x.transpose(1, 2); x_t = block_tgt.mlp_dwconv(x_t); x = x_t.transpose(1, 2)
    if hasattr(block_tgt, "mlp_transform_norm"): x = block_tgt.mlp_transform_norm(x)

    W1 = vqc_gen_fc1(t_fc1).to(dtype=x.dtype, device=x.device)
    W2 = vqc_gen_fc2(t_fc2).to(dtype=x.dtype, device=x.device)
    x = F.linear(x, W1); x = F.gelu(x); x = F.linear(x, W2); x = residual + x
    return x


def student_forward_with_vqc(vision_model, pixel_values, vqc_gens, teacher_weights):
    """Full ViT forward: blocks 22/24 use VQC-generated MLP weights."""
    x, h, w = vision_model.patch_embed(pixel_values)
    N_pos = h * w; pos_ids = torch.arange(N_pos, device=x.device)
    x = x + vision_model.pos_embed(pos_ids).unsqueeze(0)

    n_blocks = len(vision_model.blocks); x_penultimate = None
    use_concat = getattr(vision_model, 'use_concat_penultimate', False)

    for i in range(n_blocks):
        if use_concat and i == n_blocks - 2: x_penultimate = x
        if i in TARGET_BLOCKS:
            x = _vqc_mlp_block(x, vision_model.blocks[i], vision_model.blocks[SOURCE_BLOCKS[i]],
                               vqc_gens[f"fc1_{i}"], vqc_gens[f"fc2_{i}"],
                               teacher_weights[f"fc1_{i}"], teacher_weights[f"fc2_{i}"])
        else:
            x = vision_model.blocks[i](x)

    x_final = vision_model.merger(x, h, w)
    if use_concat and x_penultimate is not None:
        x_pen = vision_model.merger(x_penultimate, h, w)
        return torch.cat([x_pen, x_final], dim=-1)
    return x_final


# ═══════════════════════════════════════════════════════════════
#  Distillation
# ═══════════════════════════════════════════════════════════════

def distill(teacher_dir: Path, student_dir: Path, out_dir: Path) -> dict:
    log("=" * 60)
    log("  Dual-VQC Full-Forward Distillation (8q/6l, VQC+Hyper only)")
    log("=" * 60)

    # ── Noise images ──
    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    noise_images = [(torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=g) * 0.5 + 0.25).to(torch.bfloat16) for _ in range(N_IMAGES)]
    log(f"  Noise images: {N_IMAGES}")

    # ── Precompute teacher ViT outputs ──
    log("  Loading teacher + computing ViT outputs...")
    sys.path.insert(0, str(teacher_dir.resolve()))
    from transformers import AutoModel
    teacher = AutoModel.from_pretrained(str(teacher_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    teacher.eval()

    teacher_vit_outputs = []
    t0 = time.time()
    for i, img in enumerate(noise_images):
        with torch.no_grad(): out = teacher.vision_model(img.unsqueeze(0).to(DEVICE))
        teacher_vit_outputs.append(out.detach().cpu())
        if (i + 1) % 100 == 0: log(f"    {i+1}/{N_IMAGES} ({time.time()-t0:.0f}s)")
    log(f"    Done: {len(teacher_vit_outputs)} outputs in {time.time()-t0:.0f}s")
    del teacher; gc.collect(); torch.cuda.empty_cache()

    # ── Load teacher weights ──
    tw = {}
    teacher_sf = teacher_dir / "model.safetensors"
    with safe_open(str(teacher_sf), framework="pt", device="cpu") as f:
        for blk in TARGET_BLOCKS:
            tw[f"fc1_{blk}"] = f.get_tensor(f"vision_model.blocks.{blk}.mlp.linear_fc1.weight").to(DEVICE).to(torch.float32)
            tw[f"fc2_{blk}"] = f.get_tensor(f"vision_model.blocks.{blk}.mlp.linear_fc2.weight").to(DEVICE).to(torch.float32)

    fc1_shape = tuple(tw["fc1_22"].shape); fc2_shape = tuple(tw["fc2_22"].shape)

    # ── Load Phase 1 VQCs ──
    vqc_gens = {}
    for blk in TARGET_BLOCKS:
        for wt, shape, pd_val, p1_suffix in [("fc1", fc1_shape, PD, ""), ("fc2", fc2_shape, PD_FC2, "_fc2")]:
            gen = VQCWeightGenerator(N_WIRES, N_QLAYERS, shape, RANK, LATENT, pd_val, REUPLOAD_EVERY, N_BLOCKS, STAT_DIM, ENC_HIDDEN).to(DEVICE)
            sp = out_dir / f"vqc_we8q6l_b{blk}{p1_suffix}_state.pt"
            if not sp.exists():
                log(f"  ⚠️  Missing P1: {sp}"); continue
            ckpt = torch.load(sp, map_location=DEVICE, weights_only=True)
            state = {k: v for k, v in ckpt["state"].items() if not k.startswith("vqcs.") or "q_device" not in k}
            gen.load_state_dict(state, strict=False)
            vqc_gens[f"{wt}_{blk}"] = gen
            log(f"  Loaded {wt}_b{blk}: cos={ckpt.get('cos', '?'):.4f}")

    if len(vqc_gens) < 4:
        log("  ❌ Not all VQC generators loaded. Run Phase 1 first."); return {"status": "skipped"}

    # ── Freeze encoders ──
    for gen in vqc_gens.values():
        for p in gen.encoder.parameters(): p.requires_grad = False

    # ── Load student (frozen) ──
    log("  Loading student...")
    sys.path.insert(0, str(student_dir.resolve()))
    student = AutoModel.from_pretrained(str(student_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    student.eval()
    for p in student.parameters(): p.requires_grad = False
    s_vit = student.vision_model
    log(f"  Student: {len(s_vit.blocks)} blocks (all frozen)")

    # ── Trainable (VQC + Hyper only) ──
    trainable = []
    for gen in vqc_gens.values():
        trainable.extend(gen.vqcs.parameters()); trainable.extend(gen.hypers.parameters())
    log(f"  Trainable (VQC+Hyper): {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=LR_VQC_HYPER, weight_decay=WEIGHT_DECAY)
    wu = N_STEPS // 10

    def lr_fn(step): return LR_VQC_HYPER * (step+1)/wu if step < wu else LR_VQC_HYPER * 0.5 * (1 + math.cos(math.pi * (step-wu)/(N_STEPS-wu)))

    t0 = time.time(); best_loss = float("inf")
    for s in range(N_STEPS):
        indices = torch.randint(0, N_IMAGES, (BATCH_SIZE,))
        img_batch = torch.cat([noise_images[i].unsqueeze(0) for i in indices], dim=0).to(DEVICE)
        y_batch = torch.cat([teacher_vit_outputs[i] for i in indices], dim=0).to(DEVICE)

        opt.param_groups[0]["lr"] = lr_fn(s); opt.zero_grad()
        pred = student_forward_with_vqc(s_vit, img_batch, vqc_gens, tw)
        loss_distill = F.mse_loss(pred, y_batch)

        loss_recon = torch.tensor(0.0, device=DEVICE)
        for blk in TARGET_BLOCKS:
            for wt in ["fc1", "fc2"]:
                gen = vqc_gens[f"{wt}_{blk}"]; t_w = tw[f"{wt}_{blk}"]
                loss_recon = loss_recon + F.mse_loss(gen(t_w), t_w)
        loss = loss_distill + LAMBDA_RECON * loss_recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0); opt.step()

        if s % LOG_EVERY == 0 or s == N_STEPS - 1:
            with torch.no_grad():
                all_p, all_y = [], []
                for i in range(0, min(N_IMAGES, 100), BATCH_SIZE):
                    end = min(i + BATCH_SIZE, 100)
                    ib = torch.cat([noise_images[j].unsqueeze(0) for j in range(i, end)], dim=0).to(DEVICE)
                    yb = torch.cat([teacher_vit_outputs[j] for j in range(i, end)], dim=0).to(DEVICE)
                    pb = student_forward_with_vqc(s_vit, ib, vqc_gens, tw)
                    all_p.append(pb); all_y.append(yb)
                pf = torch.cat(all_p, dim=0); yf = torch.cat(all_y, dim=0)
                mse_full = F.mse_loss(pf, yf).item()
                cos_vit = F.cosine_similarity(pf.flatten(), yf.flatten(), dim=0).item()

                cos_w = {}
                for blk in TARGET_BLOCKS:
                    for wt in ["fc1", "fc2"]:
                        gen = vqc_gens[f"{wt}_{blk}"]; t_w = tw[f"{wt}_{blk}"]
                        Wg = gen(t_w); c = F.cosine_similarity(Wg.flatten(), t_w.flatten(), dim=0).item()
                        cos_w[f"{wt}_{blk}"] = c
                if mse_full < best_loss: best_loss = mse_full

            elapsed = time.time() - t0
            log(f"  [{elapsed:5.0f}s] {s+1:4d}/{N_STEPS} distill={loss_distill.item():.6f} MSE={mse_full:.6f} cos_vit={cos_vit:.4f} | fc1_22={cos_w['fc1_22']:.4f} fc2_22={cos_w['fc2_22']:.4f} fc1_24={cos_w['fc1_24']:.4f} fc2_24={cos_w['fc2_24']:.4f}")

    dt = time.time() - t0

    # ── Save ──
    results = {}
    for blk in TARGET_BLOCKS:
        for wt, save_suffix in [("fc1", ""), ("fc2", "_fc2")]:
            gen = vqc_gens[f"{wt}_{blk}"]; t_w = tw[f"{wt}_{blk}"]
            with torch.no_grad():
                W_final = gen(t_w).cpu()
                cos_final = F.cosine_similarity(W_final.flatten(), t_w.cpu().flatten(), dim=0).item()
            tag = f"we8q6l_b{blk}{save_suffix}_distilled_v3"
            torch.save({"state": gen.state_dict(), "cos_weight": cos_final, "best_loss": best_loss, "block": blk, "weight_type": wt, "framework": "weightencoder_8q6l_dual_distilled_v3"}, out_dir / f"vqc_{tag}_state.pt")
            torch.save(W_final, out_dir / f"vqc_{tag}_weight.pt")
            results[f"{wt}_{blk}"] = cos_final

    log(f"\n  Done: best_MSE={best_loss:.6f} time={dt:.0f}s")
    for k, v in results.items(): log(f"    {k}: cos={v:.4f}")
    return {"status": "ok", "best_loss": best_loss, "cos": results}


def main():
    p = argparse.ArgumentParser(description="Phase 2: Dual VQC distillation")
    p.add_argument("--teacher_dir", type=Path, default=os.environ.get("VQC_TEACHER_DIR", str(PROJECT_DIR / "weights" / "RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP")))
    p.add_argument("--student_dir", type=Path, default=os.environ.get("VQC_STUDENT_DIR", str(PROJECT_DIR / "weights" / "miniViT_21_24_distilled")))
    p.add_argument("--output_dir", type=Path, default=os.environ.get("VQC_OUTPUT_DIR", str(SCRIPT_DIR / "vqc_output")))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Phase 2: Dual VQC distillation | teacher={args.teacher_dir} | student={args.student_dir} | output={args.output_dir}")
    distill(args.teacher_dir, args.student_dir, args.output_dir)
    log(f"\nDone. Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
