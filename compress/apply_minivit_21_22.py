#!/usr/bin/env python3
"""
=============================================================================
 apply_minivit.py — RiverOne-QC-4B-v2-miniViT-21-24 权重复用脚本
=============================================================================
 功能：
   - 基于已蒸馏的 v2-miniViT_distilled（已有 block 23→24 共享）
   - 对 ViT 倒数第5、6层（block 21, 22）追加 MiniViT 权重复用压缩
   - block 22 共享 block 21 的 MSA/MLP 权重，增加轻量变换矩阵
   - LayerNorm 保持独立（不共享）
   - block 23→24 已有的共享和变换矩阵保持不变

 压缩范围：
   已有: block 23 → block 24（倒数第3,4层）
   新增: block 21 → block 22（倒数第5,6层）

 使用方法：
   python3 apply_minivit.py

 输出：
   ../miniViT/ 下的完整模型权重 + 修改后的 modeling_ising_vit.py
=============================================================================
"""
from __future__ import annotations

import os
import sys
import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
# 基模型：Stage 1 蒸馏输出 (已有 block[23]→block[24] 共享)
SOURCE_MODEL_DIR = os.environ.get(
    "MINIVIT_STAGE1_DISTILLED",
    str(PROJECT_DIR / "weights" / "miniViT_23_24_distilled"),
)
OUTPUT_DIR = PROJECT_DIR / "weights" / "miniViT_21_24"

# MiniViT 配置
SOURCE_BLOCK_IDX = 21   # 新增权重复用源（倒数第6层）
TARGET_BLOCK_IDX = 22   # 新增权重复用目标（倒数第5层）
NUM_ATTN_HEADS = 16


def log(msg: str):
    print(f"[MiniViT-21-24] {msg}")


# ============================================================================
# Step 1: 加载基模型
# ============================================================================

def load_source_model():
    """加载已蒸馏的 v2-miniViT 模型。"""
    log(f"加载基模型: {SOURCE_MODEL_DIR}")

    if SOURCE_MODEL_DIR not in sys.path:
        sys.path.insert(0, SOURCE_MODEL_DIR)

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        SOURCE_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    blocks = model.vision_model.blocks
    log(f"模型加载成功，ViT blocks: {len(blocks)}")
    log(f"block[23]→block[24] 共享状态: {blocks[24].attn is blocks[23].attn}")
    log(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


# ============================================================================
# Step 2: 追加权重复用（block 21 → block 22）
# ============================================================================

def apply_weight_sharing(model: nn.Module):
    """让 block 22 共享 block 21 的 MSA/MLP 权重，并增加变换矩阵。

    block 23→24 已有共享保持不变。
    """
    blocks = model.vision_model.blocks
    src_block = blocks[SOURCE_BLOCK_IDX]
    tgt_block = blocks[TARGET_BLOCK_IDX]

    log(f"block {SOURCE_BLOCK_IDX} (源) → block {TARGET_BLOCK_IDX} (目标) [新增]")

    # ── 共享 MSA 权重 ─────────────────────────────────────
    tgt_block.attn = src_block.attn
    # ── 共享 MLP 权重 ─────────────────────────────────────
    tgt_block.mlp = src_block.mlp

    # ── Attention 变换矩阵 F1, F2 ────────────────────────
    M = NUM_ATTN_HEADS
    f1_weight = torch.eye(M) + torch.randn(M, M) * 0.01
    f2_weight = torch.eye(M) + torch.randn(M, M) * 0.01
    tgt_block.register_parameter(
        "attn_transform_F1_weight",
        nn.Parameter(f1_weight, requires_grad=False),
    )
    tgt_block.register_parameter(
        "attn_transform_F2_weight",
        nn.Parameter(f2_weight, requires_grad=False),
    )

    # ── MLP 深度卷积变换 ─────────────────────────────────
    hidden_size = src_block.mlp.linear_fc1.in_features
    kernel_size = 3
    dwconv = nn.Conv1d(
        hidden_size, hidden_size, kernel_size,
        padding=kernel_size // 2, groups=hidden_size, bias=True,
    )
    nn.init.dirac_(dwconv.weight)
    tgt_block.register_module("mlp_dwconv", dwconv)

    # ── MLP 变换归一化层 ─────────────────────────────────
    transform_norm = nn.LayerNorm(hidden_size, eps=1e-6)
    tgt_block.register_module("mlp_transform_norm", transform_norm)

    # ── 验证 ─────────────────────────────────────────────
    assert tgt_block.attn is src_block.attn, "MSA 共享失败"
    assert tgt_block.mlp is src_block.mlp, "MLP 共享失败"
    assert tgt_block.norm1 is not src_block.norm1, "norm1 不应共享"
    assert tgt_block.norm2 is not src_block.norm2, "norm2 不应共享"

    log(f"MSA 共享验证: OK")
    log(f"MLP 共享验证: OK")
    log(f"变换矩阵: F1 {list(tgt_block.attn_transform_F1_weight.shape)}, "
        f"F2 {list(tgt_block.attn_transform_F2_weight.shape)}, "
        f"dwconv [C={hidden_size}, K={kernel_size}]")

    return model


# ============================================================================
# Step 3: 进一步修改 modeling_ising_vit.py（追加 block 22 共享逻辑）
# ============================================================================

def generate_modified_vit_code(output_path: Path):
    """在已修改的 modeling_ising_vit.py 基础上追加 block 21→22 共享逻辑。

    基文件已有：_forward_block_24_shared + block[23]→block[24] init + 修改后的 forward。
    """
    src_file = Path(SOURCE_MODEL_DIR) / "modeling_ising_vit.py"
    with open(src_file) as f:
        orig_code = f.read()

    # ── 1. 在 _forward_block_24_shared 方法后插入 _forward_block_22_shared ──
    shared_forward_22 = '''

    def _forward_block_22_shared(self, x: torch.Tensor) -> torch.Tensor:
        """block 22 前向传播：共享 block 21 的 MSA/MLP 权重 + 变换矩阵。

        与 _forward_block_24_shared 结构一致，操作对象为 blocks[21]/[22]。
        """
        block_22 = self.blocks[22]
        block_21 = self.blocks[21]

        residual = x
        x = block_22.norm1(x)

        B, N, C = x.shape
        M = block_21.attn.num_heads
        H_d = block_21.attn.head_dim

        qkv = block_21.attn.qkv(x).reshape(B, N, 3, M, H_d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = torch.einsum("bmnh,bmlh->bmnl", q, k) / math.sqrt(H_d)

        if hasattr(block_22, "attn_transform_F2_weight"):
            F2 = block_22.attn_transform_F2_weight
            scores = torch.einsum("bmnl,mk->bknl", scores, F2)

        attn_weights = F.softmax(scores, dim=-1)

        attn_per_head = torch.einsum("bmnl,bklh->bmknh", attn_weights, v)
        if hasattr(block_22, "attn_transform_F1_weight"):
            F1 = block_22.attn_transform_F1_weight
            attn_per_head = torch.einsum("bmknh,km->bmknh", attn_per_head, F1)
            attn_out = attn_per_head.sum(dim=2)
        else:
            attn_out = torch.einsum("bmnl,bmlh->bmnh", attn_weights, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = block_21.attn.proj(attn_out)
        x = residual + attn_out

        residual = x
        x = block_22.norm2(x)

        if hasattr(block_22, "mlp_dwconv"):
            x_t = x.transpose(1, 2)
            x_t = block_22.mlp_dwconv(x_t)
            x = x_t.transpose(1, 2)

        if hasattr(block_22, "mlp_transform_norm"):
            x = block_22.mlp_transform_norm(x)

        x = block_21.mlp(x)
        x = residual + x

        return x
'''

    # 插入位置：在 _forward_block_24_shared 方法末尾，forward(self, pixel_values) 之前
    insert_after = "        return x\n\n    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:"
    if insert_after in orig_code:
        new_method = "        return x\n" + shared_forward_22 + "\n    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:"
        orig_code = orig_code.replace(insert_after, new_method)

    # ── 2. 修改 __init__：追加 block 21→22 权重复用初始化 ──
    init_inject_22 = '''
        # ── MiniViT 权重复用 block 21→22（自动生成）───────
        _src2 = self.blocks[21]
        _tgt2 = self.blocks[22]
        _tgt2.attn = _src2.attn
        _tgt2.mlp  = _src2.mlp
        _tgt2.register_parameter("attn_transform_F1_weight",
            _torch.nn.Parameter(_torch.eye(M), requires_grad=False))
        _tgt2.register_parameter("attn_transform_F2_weight",
            _torch.nn.Parameter(_torch.eye(M), requires_grad=False))
        _dwconv2 = _torch.nn.Conv1d(H, H, 3, padding=1, groups=H)
        _torch.nn.init.dirac_(_dwconv2.weight)
        _tgt2.add_module("mlp_dwconv", _dwconv2)
        _tgt2.add_module("mlp_transform_norm",
            _torch.nn.LayerNorm(H, eps=config.layer_norm_eps))
'''

    # 插入位置：在已有 block 23→24 初始化代码之后
    # 已有代码以 `_torch.nn.LayerNorm(H, eps=config.layer_norm_eps))` 结束
    insert_marker_22 = "_torch.nn.LayerNorm(H, eps=config.layer_norm_eps))"
    if insert_marker_22 in orig_code:
        orig_code = orig_code.replace(insert_marker_22, insert_marker_22 + init_inject_22, 1)

    # ── 3. 修改 forward：block 22 走共享路径 ──────────────
    # 在现有的 `if i == 24:` 之前添加 `if i == 22:` 分支
    old_forward_if = "            if i == 24:\n                x = self._forward_block_24_shared(x)\n            else:\n                x = block(x)"
    new_forward_if = (
        "            if i == 22:\n"
        "                x = self._forward_block_22_shared(x)\n"
        "            elif i == 24:\n"
        "                x = self._forward_block_24_shared(x)\n"
        "            else:\n"
        "                x = block(x)"
    )
    if old_forward_if in orig_code:
        orig_code = orig_code.replace(old_forward_if, new_forward_if)

    output_path.write_text(orig_code, encoding="utf-8")
    log(f"已生成修改后的 modeling_ising_vit.py → {output_path}")


# ============================================================================
# Step 4: 保存模型权重
# ============================================================================

def save_model(model: nn.Module, output_dir: Path):
    """保存 MiniViT 模型权重。

    ★ 关键：不能直接用 model.state_dict()！因为 block[21]→block[22] 和
    block[23]→block[24] 共享权重后，state_dict 中会有共享内存的重复张量键，
    safetensors 会拒绝保存。
    正确做法：从基模型的 safetensors 文件直接读取所有张量，修改后写入新文件。
    """
    log(f"保存模型到 {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = Path(SOURCE_MODEL_DIR)

    # ── 1. 从基模型 safetensors 文件直接读取所有张量 ──────
    all_tensors = {}
    src_index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    src_shards = sorted(set(src_index["weight_map"].values()))
    log(f"  读取基模型 {len(src_shards)} 个分片...")
    for shard_name in src_shards:
        shard_path = src_dir / shard_name
        if not shard_path.exists():
            log(f"    ⚠️ 跳过不存在的分片: {shard_path}")
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
    log(f"  从基文件读取 {len(all_tensors)} 个张量键")

    # ── 2. 移除 block 22 的冗余 MSA/MLP 权重（block[22] 共享 block[21]）──
    prefix_22 = f"vision_model.blocks.{TARGET_BLOCK_IDX}."
    keys_to_remove = [
        k for k in all_tensors
        if k.startswith(prefix_22)
        and (".attn.qkv." in k or ".attn.proj." in k or ".mlp." in k)
    ]
    for key in keys_to_remove:
        del all_tensors[key]
    log(f"  移除 block {TARGET_BLOCK_IDX} 的 {len(keys_to_remove)} 个冗余权重键")

    # ── 3. 添加 block 22 的 MiniViT 变换矩阵 ──────────────
    tgt_block = model.vision_model.blocks[TARGET_BLOCK_IDX]

    for param_name in ["attn_transform_F1_weight", "attn_transform_F2_weight"]:
        tensor = getattr(tgt_block, param_name).data.cpu().contiguous()
        all_tensors[f"{prefix_22}{param_name}"] = tensor

    all_tensors[f"{prefix_22}mlp_dwconv.weight"] = tgt_block.mlp_dwconv.weight.data.cpu().contiguous()
    all_tensors[f"{prefix_22}mlp_dwconv.bias"] = tgt_block.mlp_dwconv.bias.data.cpu().contiguous()
    all_tensors[f"{prefix_22}mlp_transform_norm.weight"] = tgt_block.mlp_transform_norm.weight.data.cpu().contiguous()
    all_tensors[f"{prefix_22}mlp_transform_norm.bias"] = tgt_block.mlp_transform_norm.bias.data.cpu().contiguous()

    # block 22 的 norm1, norm2（独立参数，从基文件中已有，但需确认）
    # 这些键在基文件中已存在（未移除），保持不变即可

    log(f"  添加 6 个变换矩阵参数")

    # ── 4. 写入新 safetensors 分片 ────────────────────────
    max_shard_size = 2 * 1024 * 1024 * 1024
    shard = {}
    shard_idx = 0
    current_size = 0
    weight_map = {}

    for key, tensor in all_tensors.items():
        tensor = tensor.contiguous()
        ts = tensor.numel() * tensor.element_size()
        if current_size + ts > max_shard_size and shard:
            fname = f"model-{shard_idx + 1:05d}-of-00000.safetensors"
            save_file(shard, str(output_dir / fname))
            for k in shard:
                weight_map[k] = fname
            shard = {}
            shard_idx += 1
            current_size = 0
        shard[key] = tensor
        current_size += ts

    if shard:
        fname = f"model-{shard_idx + 1:05d}-of-00000.safetensors"
        save_file(shard, str(output_dir / fname))
        for k in shard:
            weight_map[k] = fname
        shard_idx += 1

    total_shards = shard_idx
    for i in range(total_shards):
        old = output_dir / f"model-{i + 1:05d}-of-00000.safetensors"
        new = output_dir / f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
            for k, v in list(weight_map.items()):
                if v == f"model-{i + 1:05d}-of-00000.safetensors":
                    weight_map[k] = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    log(f"  已保存 {total_shards} 个权重分片，{len(weight_map)} 个键")

    # ── 5. 复制配置文件 ───────────────────────────────────
    files_to_copy = [
        "config.json", "generation_config.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json",
        "configuration_riverone_qc.py", "modeling_riverone_qc.py",
        "conversation.py",
        "preprocessor_config.json", "processor_config.json",
        "chat_template.jinja", "video_preprocessor_config.json",
        "miniViT_config.json",
    ]
    for filename in files_to_copy:
        src = src_dir / filename
        dst = output_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # ── 6. 生成进一步修改的 modeling_ising_vit.py ─────────
    generate_modified_vit_code(output_dir / "modeling_ising_vit.py")

    # ── 7. 更新 miniViT_config.json ───────────────────────
    minivit_config = {
        "compression_method": "MiniViT × 2 pairs",
        "shared_pairs": [
            {"source": 21, "target": 22, "desc": "倒数第5,6层"},
            {"source": 23, "target": 24, "desc": "倒数第3,4层"},
        ],
        "shared_components": ["MSA (qkv + proj)", "MLP (fc1 + fc2)"],
        "independent_components": ["norm1", "norm2"],
        "base_model": SOURCE_MODEL_DIR,
    }
    with open(output_dir / "miniViT_config.json", "w") as f:
        json.dump(minivit_config, f, indent=2)

    log(f"  模型已完整保存到: {output_dir}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    log("=" * 60)
    log(" RiverOne-QC-4B-v2-miniViT-21-24 权重复用压缩")
    log(f" 基模型: {SOURCE_MODEL_DIR}")
    log(f" 输出目录: {OUTPUT_DIR}")
    log(f" 已有: block 23 → 24")
    log(f" 新增: block {SOURCE_BLOCK_IDX} → {TARGET_BLOCK_IDX}")
    log("=" * 60)

    model = load_source_model()
    model = apply_weight_sharing(model)
    save_model(model, OUTPUT_DIR)

    log("=" * 60)
    log(" 完成！下一步：")
    log(f"  1. 验证: python3 {SCRIPT_DIR / 'verify_minivit.py'}")
    log(f"  2. 蒸馏: python3 {SCRIPT_DIR / 'distill_minivit.py'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
