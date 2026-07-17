#!/usr/bin/env python3
"""
=============================================================================
 distill_minivit.py — RiverOne-QC-4B-v2-miniViT-21-24 权重蒸馏训练脚本
=============================================================================
 蒸馏损失（5 项）：

   L_total = L_pred + L_attn + L_hddn + L_CE-layer

 其中 L_CE-layer = α1·L_ce1 + α2·L_ce2 + α3·L_ce3    (α1=0.7, α2=0.3, α3=0.7)

   L_ce1 = CE(Teacher.block[24], Student.block[24])  ← 已有共享层（冻结）
   L_ce2 = CE(Teacher.block[26], Student.block[25])  ← 末端对齐
   L_ce3 = CE(Teacher.block[22], Student.block[22])  ← 新增共享层（可训练）

 可训练参数：
   - block[22] 的变换矩阵: F1(16×16), F2(16×16), dwconv(1152×3),
     transform_norm(1152), norm1(1152), norm2(1152)  ≈ 12K
   - block[24] 已有变换矩阵: ★ 冻结

 使用方法：
   python3 distill_minivit.py [--epochs 10] [--batch-size 4] [--lr 1e-3]
=============================================================================
"""
from __future__ import annotations

import sys, os, json, math, gc, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_DIR = os.environ.get(
    "MINIVIT_TEACHER_MODEL",
    str(PROJECT_DIR / "weights" / "RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP"),
)                                          # 原始教师模型
MINIVIT_DIR = str(PROJECT_DIR / "weights" / "miniViT_21_24")     # 权重复用后模型（student）
OUTPUT_DIR = PROJECT_DIR / "weights" / "miniViT_21_24_distilled" # 蒸馏输出
DATA_DIR = os.environ.get(
    "MINIVIT_DATA_DIR",
    "/home/lxy/workspace/datasets/vqa_format",
)
IMAGE_DIR = Path(DATA_DIR) / "images"
TRAIN_JSON = Path(DATA_DIR) / "train.json"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 10

# MiniViT 配置
SRC_22, TGT_22 = 21, 22   # 新增共享对
SRC_24, TGT_24 = 23, 24   # 已有共享对

# CE Loss 权重
ALPHA1 = 0.7   # L_ce1: T.b[24] ↔ S.b[24]（已有共享层）
ALPHA2 = 0.3   # L_ce2: T.b[26] ↔ S.b[25]（末端层）
ALPHA3 = 0.7   # L_ce3: T.b[22] ↔ S.b[22]（新增共享层）


def log(msg: str):
    print(f"[Distill-21-24] {msg}")


# ============================================================================
# 模型加载
# ============================================================================

def _clear_modeling_cache():
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "modeling_riverone_qc", "modeling_ising_vit",
            "configuration_riverone_qc", "conversation",
        ):
            del sys.modules[mod_name]


def _load_aqlm_for_vit(model, model_dir: str):
    """为 LLM 加载 AQLM 量化权重。蒸馏只用到 ViT，但需保证 LLM 权重正确以防 OOM。"""
    from collections import defaultdict

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
        if k.endswith(".codebooks"):
            grp[k[:-10]]["cb"] = k
        elif k.endswith(".codes"):
            grp[k[:-6]]["cd"] = k
        elif k.endswith(".scales"):
            grp[k[:-7]]["sc"] = k

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


def load_teacher():
    """加载原始 RiverOne-QC-4B-v2 作为 Teacher。"""
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
    log(f"Teacher 加载完成，总参数: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model.to(DEVICE)


def load_student():
    """加载 MiniViT 双重共享模型（Student），仅解冻 block[22] 参数。"""
    _clear_modeling_cache()
    while MINIVIT_DIR in sys.path:
        sys.path.remove(MINIVIT_DIR)
    sys.path.insert(0, MINIVIT_DIR)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MINIVIT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    # ★ 加载 AQLM 量化权重（关键：否则随机初始化 4.8B 参数会 OOM）
    _load_aqlm_for_vit(model, MINIVIT_DIR)

    # 冻结全部
    for p in model.parameters():
        p.requires_grad = False

    # ★ 仅解冻 block[22] 的可训练参数
    tgt_block = model.vision_model.blocks[TGT_22]

    for name in ["attn_transform_F1_weight", "attn_transform_F2_weight"]:
        if hasattr(tgt_block, name):
            getattr(tgt_block, name).requires_grad = True

    for mod_name in ["mlp_dwconv", "mlp_transform_norm"]:
        if hasattr(tgt_block, mod_name):
            for p in getattr(tgt_block, mod_name).parameters():
                p.requires_grad = True

    for norm_name in ["norm1", "norm2"]:
        for p in getattr(tgt_block, norm_name).parameters():
            p.requires_grad = True

    trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_numel = sum(p.numel() for p in model.parameters())
    log(f"Student 可训练参数: {trainable_numel:,} / 总: {total_numel:,}")
    return model.to(DEVICE)


# ============================================================================
# 图像数据
# ============================================================================

def load_image_paths() -> list:
    with open(TRAIN_JSON, "r") as f:
        data = json.load(f)
    paths = []
    for item in data:
        img_rel = item.get("image", "")
        if img_rel:
            paths.append(IMAGE_DIR / Path(img_rel).name)
    log(f"加载 {len(paths)} 张图像路径")
    return paths


_image_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_image_batch(paths: list, indices: list, device: str) -> torch.Tensor:
    batch = []
    for idx in indices:
        img = Image.open(paths[idx]).convert("RGB")
        batch.append(_image_transform(img))
    return torch.stack(batch).to(device=device, dtype=torch.bfloat16)


# ============================================================================
# 共享块前向（手动实现，不依赖 modeling 文件）
# ============================================================================

def _shared_block_forward(x: torch.Tensor, vit, src_idx: int, tgt_idx: int) -> torch.Tensor:
    """通用共享块前向：tgt 块使用 src 块的权重 + tgt 块上的变换矩阵。"""
    block_tgt = vit.blocks[tgt_idx]
    block_src = vit.blocks[src_idx]

    residual = x
    x = block_tgt.norm1(x)
    B, N, C = x.shape
    M = block_src.attn.num_heads
    H_d = block_src.attn.head_dim

    qkv = block_src.attn.qkv(x).reshape(B, N, 3, M, H_d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H_d)

    if hasattr(block_tgt, "attn_transform_F2_weight"):
        scores = torch.einsum("bmnl,mk->bknl", scores, block_tgt.attn_transform_F2_weight)

    attn_w = F.softmax(scores, dim=-1)
    attn_ph = torch.einsum("bmnl,bklh->bmknh", attn_w, v)
    if hasattr(block_tgt, "attn_transform_F1_weight"):
        attn_ph = torch.einsum("bmknh,km->bmknh", attn_ph, block_tgt.attn_transform_F1_weight)
        attn_o = attn_ph.sum(dim=2)
    else:
        attn_o = torch.einsum("bmnl,bmlh->bmnh", attn_w, v)
    attn_o = attn_o.transpose(1, 2).reshape(B, N, C)
    attn_o = block_src.attn.proj(attn_o)
    x = residual + attn_o

    residual = x
    x = block_tgt.norm2(x)
    if hasattr(block_tgt, "mlp_dwconv"):
        x_t = x.transpose(1, 2)
        x_t = block_tgt.mlp_dwconv(x_t)
        x = x_t.transpose(1, 2)
    if hasattr(block_tgt, "mlp_transform_norm"):
        x = block_tgt.mlp_transform_norm(x)
    x = block_src.mlp(x)
    x = residual + x
    return x


# ============================================================================
# Teacher Forward
# ============================================================================

def forward_teacher_block24(teacher_model, pixel_values):
    """Teacher ViT forward，收集 block 24 输出 + merger 特征 + attn weights。"""
    vit = teacher_model.vision_model
    stored_hidden = {}
    stored_attn = {}

    def hook_hidden(module, input, output):
        stored_hidden["b24"] = output.detach()

    def hook_attn(module, input, output):
        x = input[0]
        B, N, C2 = x.shape
        M = module.num_heads
        H = C2 // M
        qkv = module.qkv(x).reshape(B, N, 3, M, H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H)
        stored_attn["b24_attn"] = F.softmax(scores, dim=-1).detach()

    h1 = vit.blocks[TGT_24].register_forward_hook(hook_hidden)
    h2 = vit.blocks[TGT_24].attn.register_forward_hook(hook_attn)

    with torch.no_grad():
        vit_output = vit(pixel_values)

    h1.remove(); h2.remove()
    return vit_output.detach(), stored_hidden["b24"], stored_attn["b24_attn"]


def capture_teacher_hiddens(teacher_model, pixel_values, layer_idxs: list) -> dict:
    """捕获 Teacher 指定层的 hidden states。"""
    vit = teacher_model.vision_model
    stored = {}
    handles = []
    for idx in layer_idxs:
        d = {}
        def make_hook(d):
            def hook(module, input, output):
                d["h"] = output.detach()
            return hook
        h = vit.blocks[idx].register_forward_hook(make_hook(d))
        handles.append(h)
        stored[idx] = d
    with torch.no_grad():
        vit(pixel_values)
    for h in handles:
        h.remove()
    return {k: d["h"] for k, d in stored.items()}


# ============================================================================
# Student Forward（全手动，不依赖 modeling 文件的自动注入）
# ============================================================================

def student_full_forward(student_vision, pixel_values):
    """完整 student forward，捕获 block[22], block[24], block[25] hidden states。"""
    x, h, w = student_vision.patch_embed(pixel_values)
    N_pos = h * w
    pos_ids = torch.arange(N_pos, device=x.device)
    x = x + student_vision.pos_embed(pos_ids).unsqueeze(0)

    hidden_22 = None
    hidden_24 = None
    hidden_25 = None
    x_penultimate = None

    n_blocks = len(student_vision.blocks)
    for i, block in enumerate(student_vision.blocks):
        if i == 21:
            x = block(x)
            continue
        if i == 22:
            x = _shared_block_forward(x, student_vision, SRC_22, TGT_22)
            hidden_22 = x
            continue
        if i == 23:
            x = block(x)
            continue
        if i == 24:
            x = _shared_block_forward(x, student_vision, SRC_24, TGT_24)
            hidden_24 = x
            continue
        if i == 25:
            if student_vision.use_concat_penultimate:
                x_penultimate = x  # block[24] output
            x = block(x)
            hidden_25 = x
            continue

        if student_vision.use_concat_penultimate and (i == n_blocks - 2):
            x_penultimate = x
        x = block(x)

    x_final = student_vision.merger(x, h, w)
    if student_vision.use_concat_penultimate and x_penultimate is not None:
        x_pen = student_vision.merger(x_penultimate, h, w)
        merger_output = torch.cat([x_pen, x_final], dim=-1)
    else:
        merger_output = x_final

    return merger_output, hidden_22, hidden_24, hidden_25


def student_block24_grad(vit, pixel_values):
    """手动跑 student 的 block 0-24（带梯度），获取 hidden_24 + attn_weights。"""
    x, h, w = vit.patch_embed(pixel_values)
    N_pos = h * w
    pos_ids = torch.arange(N_pos, device=x.device)
    x = x + vit.pos_embed(pos_ids).unsqueeze(0)

    # blocks 0-20: normal
    for i in range(21):
        x = vit.blocks[i](x)
    # block 21: normal
    x = vit.blocks[21](x)
    # block 22: shared
    x = _shared_block_forward(x, vit, SRC_22, TGT_22)
    # block 23: normal
    x = vit.blocks[23](x)

    # block 24: shared (compute attn weights too)
    block_24 = vit.blocks[TGT_24]
    block_23 = vit.blocks[SRC_24]
    x_n = block_24.norm1(x)
    B, N, C = x_n.shape
    M = block_23.attn.num_heads
    H_d = block_23.attn.head_dim
    qkv = block_23.attn.qkv(x_n).reshape(B, N, 3, M, H_d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H_d)
    if hasattr(block_24, "attn_transform_F2_weight"):
        scores = torch.einsum("bmnl,mk->bknl", scores, block_24.attn_transform_F2_weight)
    attn_weights = F.softmax(scores, dim=-1)
    hidden_24 = _shared_block_forward(x, vit, SRC_24, TGT_24)
    return hidden_24, attn_weights


# ============================================================================
# 损失函数
# ============================================================================

def layerwise_ce_loss(ht, hs, eps=1e-8):
    pt = F.softmax(ht, dim=-1)
    ps = F.log_softmax(hs, dim=-1)
    return -(pt * ps).sum(dim=-1).mean()


def compute_losses(feat_t, feat_s, hidden_t, hidden_s, attn_t, attn_s,
                   t_b22, s_b22, t_b24, s_b24, t_b26, s_b25) -> dict:
    L_pred = F.mse_loss(feat_s, feat_t)
    L_attn = F.mse_loss(attn_s, attn_t)

    def gram(x):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.bmm(x, x.transpose(1, 2))
    L_hddn = F.mse_loss(gram(hidden_s), gram(hidden_t))

    L_ce1 = layerwise_ce_loss(t_b24, s_b24)   # 已有共享层（α=0.7）
    L_ce2 = layerwise_ce_loss(t_b26, s_b25)   # 末端层（α=0.3）
    L_ce3 = layerwise_ce_loss(t_b22, s_b22)   # 新增共享层（α=0.7）
    L_ce_layer = ALPHA1 * L_ce1 + ALPHA2 * L_ce2 + ALPHA3 * L_ce3

    L_total = L_pred + L_attn + L_hddn + L_ce_layer

    return {
        "L_pred": L_pred.item(), "L_attn": L_attn.item(), "L_hddn": L_hddn.item(),
        "L_ce1": L_ce1.item(), "L_ce2": L_ce2.item(), "L_ce3": L_ce3.item(),
        "L_ce_layer": L_ce_layer.item(), "L_total": L_total,
    }


# ============================================================================
# 保存
# ============================================================================

def _save_distilled(student_model, output_dir: Path, epoch: int, losses: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(MINIVIT_DIR)
    idx = json.loads((src_dir / "model.safetensors.index.json").read_text())
    all_tensors = {}
    for shard_name in sorted(set(idx["weight_map"].values())):
        shard_path = src_dir / shard_name
        if shard_path.exists():
            all_tensors.update(load_file(str(shard_path)))

    student_state = student_model.state_dict()
    prefix = "vision_model.blocks.22."
    for key in list(student_state.keys()):
        if key.startswith(prefix) and key in all_tensors:
            all_tensors[key] = student_state[key].detach().cpu().contiguous()
    for norm_name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
        full_key = f"vision_model.blocks.22.{norm_name}"
        if full_key in all_tensors and full_key in student_state:
            all_tensors[full_key] = student_state[full_key].detach().cpu().contiguous()

    save_file(all_tensors, str(output_dir / "model.safetensors"))

    import shutil
    for fname in [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
        "configuration_riverone_qc.py", "modeling_riverone_qc.py",
        "modeling_ising_vit.py", "conversation.py",
        "preprocessor_config.json", "processor_config.json",
        "chat_template.jinja", "video_preprocessor_config.json",
        "miniViT_config.json",
    ]:
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy2(str(src_f), str(output_dir / fname))

    wm = {k: "model.safetensors" for k in all_tensors}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f, indent=2)
    log(f"  已保存蒸馏后模型到: {output_dir}")


# ============================================================================
# 蒸馏训练循环
# ============================================================================

def distill(args):
    log("=" * 60)
    log(" RiverOne-QC-4B-v2-miniViT-21-24 蒸馏训练")
    log(f" α1=0.7, α2=0.3, α3=0.7, lr={args.lr}, epochs={args.epochs}")
    log("=" * 60)

    teacher = load_teacher()
    student = load_student()
    image_paths = load_image_paths()
    num_images = len(image_paths)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        student.train()
        epoch_losses = defaultdict(float)
        steps = args.steps_per_epoch

        for step in range(steps):
            indices = torch.randint(0, num_images, (args.batch_size,)).tolist()
            images = load_image_batch(image_paths, indices, DEVICE)

            # ── Teacher ────────────────────────────────────
            feat_t, hidden_t, attn_t = forward_teacher_block24(teacher, images)
            t_hiddens = capture_teacher_hiddens(teacher, images, [22, 24, 26])
            t_b22 = t_hiddens[22]
            t_b24 = t_hiddens[24]
            t_b26 = t_hiddens[26]

            # ── Student ────────────────────────────────────
            student_vision = student.vision_model
            feat_s, s_b22, s_b24, s_b25 = student_full_forward(student_vision, images)
            hidden_s, attn_s = student_block24_grad(student_vision, images)

            # ── Losses ─────────────────────────────────────
            losses = compute_losses(
                feat_t, feat_s, hidden_t, hidden_s, attn_t, attn_s,
                t_b22, s_b22, t_b24, s_b24, t_b26, s_b25,
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
                    f"  Epoch {epoch+1}/{args.epochs} Step {step}/{steps} | "
                    f"L_pred={losses['L_pred']:.6f} L_attn={losses['L_attn']:.6f} "
                    f"L_hddn={losses['L_hddn']:.6f} "
                    f"L_ce1={losses['L_ce1']:.6f} L_ce2={losses['L_ce2']:.6f} "
                    f"L_ce3={losses['L_ce3']:.6f} L_total={losses['L_total'].item():.6f}"
                )

        scheduler.step()
        avg = {k: v / steps for k, v in epoch_losses.items()}
        avg_total = sum(avg.values())
        log(f"  Epoch {epoch+1} 平均 | Total={avg_total:.6f}")

        if avg_total < best_loss:
            best_loss = avg_total
            _save_distilled(student, OUTPUT_DIR, epoch + 1, avg)
            log(f"  ✅ 保存最佳模型 (loss={best_loss:.6f})")

    log(f"\n训练完成！最佳损失: {best_loss:.6f}")
    log(f"蒸馏后模型: {OUTPUT_DIR}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiverOne-QC-4B-v2-miniViT-21-24 蒸馏训练")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    args = parser.parse_args()
    distill(args)
