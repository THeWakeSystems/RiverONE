#!/usr/bin/env python3
"""
=============================================================================
 distill_minivit.py — MiniViT 权重蒸馏训练脚本
=============================================================================
 严格遵循 MiniViT 论文 (CVPR 2022) 第 3.3 节蒸馏方案：

   L_total = L_pred + β·L_attn + γ·L_hddn   (β=γ=1.0)

   L_attn : 对齐 Teacher/Student block 24 的 attention maps（MSE）
   L_hddn : 对齐 Teacher/Student block 24 输出的 Gram 矩阵（MSE）
   L_pred : 对齐 Teacher/Student ViT merger 输出特征（MSE）

 可训练参数（~12K）：
   - attn_transform_F1_weight, attn_transform_F2_weight (各 16×16=256)
   - mlp_dwconv.weight, mlp_dwconv.bias (1152×3 + 1152)
   - mlp_transform_norm.weight, mlp_transform_norm.bias (各 1152)
   - norm1.weight, norm1.bias, norm2.weight, norm2.bias (各 1152)

 使用方法：
   python3 distill_minivit.py [--epochs 10] [--batch-size 4] [--lr 1e-3]

 输出：
   ../miniViT_distilled/ 下的蒸馏后权重

 适配模型：
   RiverOne-QC-4B-v1（miniViT 视觉编码器 + Qwen3-4B LLM + AQLM 量化）
=============================================================================
"""
from __future__ import annotations

import sys, os, json, math, time, gc, argparse, warnings
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = (
    "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/"
    "AQLM/RiverOne-QC-4B-v2-AQLM-36L"
)
MINIVIT_DIR = str(SCRIPT_DIR.parent / "weights" / "miniViT_v2")
OUTPUT_DIR = SCRIPT_DIR.parent / "weights" / "miniViT_v2_distilled"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 10  # 每 N 步打印损失

# MiniViT 配置（与 apply_minivit.py 一致）
TARGET_BLOCK_IDX = 24
SOURCE_BLOCK_IDX = 23


def log(msg: str):
    print(f"[Distill] {msg}")


# ============================================================================
# 模型加载
# ============================================================================

def _clear_modeling_cache():
    """清除 modeling 相关模块缓存，确保从正确路径加载。"""
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "modeling_riverone_qc", "modeling_ising_vit",
            "configuration_riverone_qc", "conversation",
        ):
            del sys.modules[mod_name]


def _load_aqlm_for_vit(model, model_dir: str):
    """为 LLM 加载 AQLM 量化权重。蒸馏只用到 ViT，但需保证 LLM 权重正确以防报错。"""
    qc = Path(model_dir) / "quant_config.json"
    if not qc.exists():
        return
    if json.loads(open(qc).read()).get("quantization_method") != "AQLM":
        return

    from aqlm import QuantizedLinear as AQLMLinear

    idx = json.loads(
        (Path(model_dir) / "model.safetensors.index.json").read_text()
    )
    wm = idx["weight_map"]

    grp = defaultdict(dict)
    for k in wm:
        if k.endswith(".codebooks"): grp[k[:-10]]["cb"] = k
        elif k.endswith(".codes"):   grp[k[:-6]]["cd"] = k
        elif k.endswith(".scales"):  grp[k[:-7]]["sc"] = k

    tens = {}
    for s in sorted(set(wm.values())):
        p = Path(model_dir) / s
        if p.exists():
            tens.update(load_file(str(p)))

    layers = model.language_model.model.layers
    for base, info in grp.items():
        parts = base.split(".")
        li = int(parts[3])
        sp = parts[4:]
        cb, cd, sc = tens[info["cb"]], tens[info["cd"]], tens[info["sc"]]
        nc, cs, og, ig = cb.shape
        ql = AQLMLinear(
            cd.shape[1] * ig, cd.shape[0] * og, ig, og, nc,
            cs.bit_length() - 1, bias=False, dtype=cb.dtype,
        )
        ql.codebooks.data.copy_(cb)
        ql.codes.data.copy_(cd.to(ql.codes.dtype))
        ql.scales.data.copy_(sc)
        parent = layers[li]
        for seg in sp[:-1]:
            parent = getattr(parent, seg)
        ql = ql.to(next(parent.parameters()).device)
        setattr(parent, sp[-1], ql)


def load_teacher_vit():
    """加载原始 ViT 编码器（Teacher），冻结全部参数。"""
    _clear_modeling_cache()
    while SOURCE_DIR in sys.path:
        sys.path.remove(SOURCE_DIR)
    sys.path.insert(0, SOURCE_DIR)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        SOURCE_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    _load_aqlm_for_vit(model, SOURCE_DIR)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


def load_student_vit():
    """加载 MiniViT 压缩模型（Student），设置可训练参数。"""
    _clear_modeling_cache()
    while MINIVIT_DIR in sys.path:
        sys.path.remove(MINIVIT_DIR)
    sys.path.insert(0, MINIVIT_DIR)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MINIVIT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    _load_aqlm_for_vit(model, MINIVIT_DIR)

    # 先冻结全部
    for p in model.parameters():
        p.requires_grad = False

    # 解冻目标参数
    tgt_block = model.vision_model.blocks[TARGET_BLOCK_IDX]
    trainable_ids = []

    for name in ["attn_transform_F1_weight", "attn_transform_F2_weight"]:
        if hasattr(tgt_block, name):
            p = getattr(tgt_block, name)
            p.requires_grad = True
            trainable_ids.append(id(p))

    for mod_name in ["mlp_dwconv", "mlp_transform_norm"]:
        if hasattr(tgt_block, mod_name):
            mod = getattr(tgt_block, mod_name)
            for p in mod.parameters():
                p.requires_grad = True
                trainable_ids.append(id(p))

    # block 24 独立 LayerNorm（可训练）
    for norm_name in ["norm1", "norm2"]:
        norm = getattr(tgt_block, norm_name)
        for p in norm.parameters():
            p.requires_grad = True
            trainable_ids.append(id(p))

    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    trainable_numel = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_numel = sum(p.numel() for p in model.parameters())
    log(
        f"可训练参数: {trainable_numel:,} ({trainable_count} tensors) "
        f"/ {total_numel:,}"
    )
    return model.to(DEVICE)


# ============================================================================
# 注意力权重捕获（Teacher 端）
# ============================================================================

def _capture_teacher_attn_weights(teacher_model, pixel_values):
    """通过 hook 捕获 teacher block 24 的 attention weights。"""
    block_24 = teacher_model.vision_model.blocks[TARGET_BLOCK_IDX]
    attn_module = block_24.attn
    stored = {}

    def hook_fn(module, input, output):
        x = input[0]
        B, N, C = x.shape
        M = module.num_heads
        H = C // M
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, M, H)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H)
        stored["attn_weights"] = F.softmax(scores, dim=-1).detach()
        stored["q"] = q.detach()
        stored["k"] = k.detach()
        stored["v"] = v.detach()

    handle = attn_module.register_forward_hook(hook_fn)
    return handle, stored


# ============================================================================
# ViT Forward（Teacher 端）
# ============================================================================

def forward_teacher_vit(teacher_model, pixel_values):
    """Teacher ViT forward，收集 block 24 输出 + merger 特征 + attn weights。"""
    vit = teacher_model.vision_model

    block_24_output = {}

    def hook_block_24(module, input, output):
        block_24_output["hidden"] = output.detach()

    h = vit.blocks[TARGET_BLOCK_IDX].register_forward_hook(hook_block_24)
    attn_h, attn_data = _capture_teacher_attn_weights(
        teacher_model, pixel_values
    )

    with torch.no_grad():
        vit_output = vit(pixel_values)

    h.remove()
    attn_h.remove()

    return (
        vit_output.detach(),
        block_24_output["hidden"],
        attn_data["attn_weights"],
    )


# ============================================================================
# Student ViT Forward
# ============================================================================

def forward_student_vit(student_model, pixel_values):
    """Student ViT forward，捕获 block 24 中间量。"""
    vit = student_model.vision_model
    stored = {}

    def hook_in(module, input):
        stored["input_to_24"] = input[0].detach()

    def hook_out(module, input, output):
        stored["hidden_24"] = output.detach()

    h_in = vit.blocks[TARGET_BLOCK_IDX].register_forward_pre_hook(hook_in)
    h_out = vit.blocks[TARGET_BLOCK_IDX].register_forward_hook(hook_out)

    with torch.no_grad():
        vit_output = vit(pixel_values)

    h_in.remove()
    h_out.remove()

    # 手动计算 student block 24 attention weights
    x24 = stored["input_to_24"]
    block_24 = vit.blocks[TARGET_BLOCK_IDX]
    block_23 = vit.blocks[SOURCE_BLOCK_IDX]

    B, N, C = x24.shape
    M = block_23.attn.num_heads
    H = block_23.attn.head_dim

    x_normed = block_24.norm1(x24)
    qkv = (
        block_23.attn.qkv(x_normed)
        .reshape(B, N, 3, M, H)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H)

    if hasattr(block_24, "attn_transform_F2_weight"):
        F2 = block_24.attn_transform_F2_weight
        scores = torch.einsum("bmnl,mk->bknl", scores, F2)

    student_attn_weights = F.softmax(scores, dim=-1).detach()
    return vit_output.detach(), stored["hidden_24"], student_attn_weights


# ============================================================================
# 损失函数
# ============================================================================

def compute_losses(feat_t, feat_s, hidden_t, hidden_s, attn_t, attn_s) -> dict:
    """计算 MiniViT 蒸馏三项损失。

    Args:
        feat_t, feat_s: Teacher/Student ViT merger 输出 [B, N', C']
        hidden_t, hidden_s: block 24 输出 [B, N, C]
        attn_t, attn_s: block 24 attention weights [B, M, N, N]

    Returns:
        dict with keys: L_pred, L_attn, L_hddn, L_total
    """
    L_pred = F.mse_loss(feat_s, feat_t)
    L_attn = F.mse_loss(attn_s, attn_t)

    def gram(x):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.bmm(x, x.transpose(1, 2))

    G_t = gram(hidden_t)
    G_s = gram(hidden_s)
    L_hddn = F.mse_loss(G_s, G_t)

    L_total = L_pred + L_attn + L_hddn

    return {
        "L_pred": L_pred.item(),
        "L_attn": L_attn.item(),
        "L_hddn": L_hddn.item(),
        "L_total": L_total,
    }


# ============================================================================
# 图像数据生成
# ============================================================================

def generate_dummy_images(batch_size: int, image_size: int = 448):
    """生成随机图像批次用于蒸馏（无需标注）。"""
    return (
        torch.rand(batch_size, 3, image_size, image_size, device=DEVICE)
        * 0.5 + 0.25
    ).to(torch.bfloat16)


# ============================================================================
# Student block 24 forward（带梯度，用于训练）
# ============================================================================

def _student_block24_forward_with_grad(vit, pixel_values):
    """手动跑 student ViT 的 block 0-24，获取 block 24 hidden + attn weights（带梯度）。"""
    x, h, w = vit.patch_embed(pixel_values)
    N_pos = h * w
    pos_ids = torch.arange(N_pos, device=x.device)
    x = x + vit.pos_embed(pos_ids).unsqueeze(0)

    for i in range(SOURCE_BLOCK_IDX + 1):
        x = vit.blocks[i](x)

    block_24 = vit.blocks[TARGET_BLOCK_IDX]
    block_23 = vit.blocks[SOURCE_BLOCK_IDX]

    residual = x
    x_normed = block_24.norm1(x)

    B, N, C = x_normed.shape
    M = block_23.attn.num_heads
    H = block_23.attn.head_dim

    qkv = (
        block_23.attn.qkv(x_normed)
        .reshape(B, N, 3, M, H)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H)

    if hasattr(block_24, "attn_transform_F2_weight"):
        F2 = block_24.attn_transform_F2_weight
        scores = torch.einsum("bmnl,mk->bknl", scores, F2)

    attn_weights = F.softmax(scores, dim=-1)

    attn_per_head = torch.einsum("bmnl,bklh->bmknh", attn_weights, v)
    if hasattr(block_24, "attn_transform_F1_weight"):
        F1 = block_24.attn_transform_F1_weight
        attn_per_head = torch.einsum("bmknh,km->bmknh", attn_per_head, F1)
        attn_out = attn_per_head.sum(dim=2)
    else:
        attn_out = torch.einsum("bmnl,bmlh->bmnh", attn_weights, v)

    attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
    attn_out = block_23.attn.proj(attn_out)
    x = residual + attn_out

    residual = x
    x = block_24.norm2(x)

    if hasattr(block_24, "mlp_dwconv"):
        x_t = x.transpose(1, 2)
        x_t = block_24.mlp_dwconv(x_t)
        x = x_t.transpose(1, 2)

    if hasattr(block_24, "mlp_transform_norm"):
        x = block_24.mlp_transform_norm(x)

    x = block_23.mlp(x)
    hidden_24 = residual + x

    return hidden_24, attn_weights


# ============================================================================
# 蒸馏后模型保存
# ============================================================================

def _save_distilled_model(
    student_model, output_dir: Path, epoch: int, losses: dict
):
    """保存蒸馏后的 MiniViT 权重。

    复制源 MiniViT 的全部文件，替换可训练参数。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取源 MiniViT 权重索引
    src_dir = Path(MINIVIT_DIR)
    idx = json.loads(
        (src_dir / "model.safetensors.index.json").read_text()
    )
    all_tensors = {}
    for shard_name in sorted(set(idx["weight_map"].values())):
        shard_path = src_dir / shard_name
        if shard_path.exists():
            all_tensors.update(load_file(str(shard_path)))

    # 2. 用 student 的可训练参数覆盖
    student_state = student_model.state_dict()
    prefix = "vision_model.blocks.24."

    for key in list(student_state.keys()):
        if key.startswith(prefix):
            sub_key = key[len(prefix):]
            full_key = f"vision_model.blocks.24.{sub_key}"
            if full_key in all_tensors:
                all_tensors[full_key] = (
                    student_state[key].detach().cpu().contiguous()
                )
            elif key in all_tensors:
                all_tensors[key] = (
                    student_state[key].detach().cpu().contiguous()
                )

    # 3. 也更新 norm1, norm2（可能在不同分片中）
    for norm_name in [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"
    ]:
        full_key = f"vision_model.blocks.24.{norm_name}"
        student_key = f"vision_model.blocks.24.{norm_name}"
        if student_key in student_state and full_key in all_tensors:
            all_tensors[full_key] = (
                student_state[student_key].detach().cpu().contiguous()
            )

    # 4. 保存
    save_file(all_tensors, str(output_dir / "model.safetensors"))

    # 5. 复制其他文件
    import shutil
    for fname in [
        "config.json", "generation_config.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json",
        "configuration_riverone_qc.py", "modeling_riverone_qc.py",
        "modeling_ising_vit.py", "conversation.py",
        "preprocessor_config.json", "processor_config.json",
        "chat_template.jinja", "video_preprocessor_config.json",
        "quant_config.json", "miniViT_config.json",
    ]:
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy2(str(src_f), str(output_dir / fname))

    # 6. 更新权重映射
    wm = {}
    for k in all_tensors:
        wm[k] = "model.safetensors"
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f, indent=2)

    log(f"  已保存蒸馏后模型到: {output_dir}")


# ============================================================================
# 蒸馏训练循环
# ============================================================================

def distill(args):
    log("=" * 60)
    log(" MiniViT 权重蒸馏训练（Phase 2）")
    log(f" β={args.beta}, γ={args.gamma}, lr={args.lr}, epochs={args.epochs}")
    log("=" * 60)

    # 1. 加载模型
    log("加载 Teacher（原始 ViT）...")
    teacher = load_teacher_vit()
    log("加载 Student（MiniViT）...")
    student = load_student_vit()

    # 2. 优化器
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # 3. 训练循环
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    loss_history = []

    for epoch in range(args.epochs):
        student.train()
        epoch_losses = defaultdict(float)
        steps = args.steps_per_epoch

        for step in range(steps):
            images = generate_dummy_images(args.batch_size)

            # Teacher forward
            feat_t, hidden_t, attn_t = forward_teacher_vit(teacher, images)

            # Student forward（带梯度）
            student_vision = student.vision_model
            feat_s_raw = student_vision(images)
            feat_s = feat_s_raw

            hidden_s, attn_s = _student_block24_forward_with_grad(
                student_vision, images
            )

            # 损失计算
            losses = compute_losses(
                feat_t, feat_s, hidden_t, hidden_s, attn_t, attn_s
            )

            optimizer.zero_grad()
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                if k != "L_total":
                    epoch_losses[k] += v

            if step % LOG_EVERY == 0:
                log(
                    f"  Epoch {epoch+1}/{args.epochs} "
                    f"Step {step}/{steps} | "
                    f"L_pred={losses['L_pred']:.6f} "
                    f"L_attn={losses['L_attn']:.6f} "
                    f"L_hddn={losses['L_hddn']:.6f} "
                    f"L_total={losses['L_total'].item():.6f}"
                )

        scheduler.step()

        # Epoch 摘要
        avg = {k: v / steps for k, v in epoch_losses.items()}
        avg_total = sum(avg.values())
        loss_history.append(avg_total)
        log(
            f"  Epoch {epoch+1} 平均 | L_pred={avg['L_pred']:.6f} "
            f"L_attn={avg['L_attn']:.6f} L_hddn={avg['L_hddn']:.6f} "
            f"Total={avg_total:.6f}"
        )

        # 保存最佳
        if avg_total < best_loss:
            best_loss = avg_total
            _save_distilled_model(student, OUTPUT_DIR, epoch + 1, avg)
            log(f"  ✅ 保存最佳模型 (loss={best_loss:.6f})")

    log(f"\n训练完成！最佳损失: {best_loss:.6f}")
    log(f"蒸馏后模型: {OUTPUT_DIR}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniViT 权重蒸馏")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    args = parser.parse_args()
    distill(args)
