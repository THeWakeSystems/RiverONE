#!/usr/bin/env python3
"""
=============================================================================
 apply_minivit.py — RiverOne-QC-4B-v1-AQLM-36L 视觉编码器 MiniViT 权重复用脚本
=============================================================================
 功能：
   - 加载已完成 AQLM 量化的 RiverOne-QC-4B-v1-AQLM-36L
   - 对 ViT 倒数第3、4层（block 23, 24）应用 MiniViT 权重复用压缩
   - block 24 共享 block 23 的 MSA/MLP 权重，增加轻量变换矩阵
   - LayerNorm 保持独立（不共享）
   - LLM 分支完全不动（已量化部分不受影响）

 MiniViT 论文原理 (CVPR 2022)：
   - 相邻 Transformer 层的 MSA/MLP 权重高度相似，可共享
   - 通过轻量变换矩阵（F1, F2, dwconv）补偿共享带来的精度损失
   - 总新增参数量 ~12K，远小于节省的参数量（~14M per layer）

 使用方法：
   python3 apply_minivit.py

 输出：
   ../miniViT/ 下的完整模型权重 + 修改后的 modeling_ising_vit.py

 与参考实现的关系：
   参考: RiverOne-QC-4B-AQLM-36L-last34-miniViT/scripts/apply_minivit.py
   本脚本适配: RiverOne-QC-4B-v1（miniViT 视觉编码器版本）
=============================================================================
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import copy
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SOURCE_MODEL_DIR = (
    "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/"
    "AQLM/RiverOne-QC-4B-v2-AQLM-36L"
)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "weights" / "miniViT_v2"  # 权重输出到 weights/miniViT_v2/

# MiniViT 配置
SOURCE_BLOCK_IDX = 23   # 权重复用源（倒数第4层，0-indexed）
TARGET_BLOCK_IDX = 24   # 权重复用目标（倒数第3层）
NUM_ATTN_HEADS = 16     # Ising ViT 注意力头数（hidden=1152→head_dim=72）


def log(msg: str):
    print(f"[MiniViT] {msg}")


# ============================================================================
# Step 1: 加载源模型（使用 AutoModel 避免相对导入问题）
# ============================================================================

def load_source_model():
    """加载源模型。

    使用 AutoModel.from_pretrained 加载模型结构（ViT 权重正确加载），
    然后通过 _load_aqlm 将 LLM 的 Linear 层替换为 AQLM QuantizedLinear，
    确保内存中的模型权重完整正确。
    """
    log(f"加载源模型: {SOURCE_MODEL_DIR}")

    # 将源模型目录加入 sys.path
    if SOURCE_MODEL_DIR not in sys.path:
        sys.path.insert(0, SOURCE_MODEL_DIR)

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        SOURCE_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # 加载 AQLM 量化权重（替换 LLM Linear 层为 QuantizedLinear）
    _load_aqlm(model, SOURCE_MODEL_DIR)

    log(f"模型加载成功，num_image_token={model.num_image_token}")
    log(f"ViT blocks 数量: {len(model.vision_model.blocks)}")
    log(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def _load_aqlm(model, model_dir: str):
    """加载 AQLM 量化权重到 LLM 的 Linear 层。

    从 safetensors 中读取 .codebooks/.codes/.scales 张量，
    创建 AQLM QuantizedLinear 层并替换原始 Linear 层。
    仅影响 LLM 分支，ViT 分支保持不变。
    """
    from collections import defaultdict

    qc = Path(model_dir) / "quant_config.json"
    if not qc.exists():
        log("  未找到 quant_config.json，跳过 AQLM 加载")
        return

    quant_cfg = json.loads(open(qc).read())
    if quant_cfg.get("quantization_method") != "AQLM":
        log(f"  量化方法非 AQLM，跳过")
        return

    from aqlm import QuantizedLinear as AQLMLinear

    idx = json.loads(
        (Path(model_dir) / "model.safetensors.index.json").read_text()
    )
    wm = idx["weight_map"]

    # 按前缀分组：将 .codebooks/.codes/.scales 归到同一个 Linear 层
    grp = defaultdict(dict)
    for k in wm:
        if k.endswith(".codebooks"):
            grp[k[:-10]]["cb"] = k
        elif k.endswith(".codes"):
            grp[k[:-6]]["cd"] = k
        elif k.endswith(".scales"):
            grp[k[:-7]]["sc"] = k

    # 加载所有分片
    tens = {}
    for s in sorted(set(wm.values())):
        p = Path(model_dir) / s
        if p.exists():
            from safetensors import safe_open
            with safe_open(str(p), framework="pt", device="cpu") as f:
                for k in f.keys():
                    tens[k] = f.get_tensor(k)

    layers = model.language_model.model.layers
    replaced = 0
    for base, info in grp.items():
        parts = base.split(".")
        li = int(parts[3])  # layers.X
        sp = parts[4:]       # self_attn.q_proj etc.

        cb = tens[info["cb"]]
        cd = tens[info["cd"]]
        sc = tens[info["sc"]]

        nc, cs, og, ig = cb.shape
        ql = AQLMLinear(
            cd.shape[1] * ig,  # in_features
            cd.shape[0] * og,  # out_features
            ig, og, nc,
            cs.bit_length() - 1,
            bias=False,
            dtype=cb.dtype,
        )
        ql.codebooks.data.copy_(cb)
        ql.codes.data.copy_(cd.to(ql.codes.dtype))
        ql.scales.data.copy_(sc)

        parent = layers[li]
        for seg in sp[:-1]:
            parent = getattr(parent, seg)
        ql = ql.to(next(parent.parameters()).device)
        setattr(parent, sp[-1], ql)
        replaced += 1

    log(f"  AQLM 加载完成: 替换了 {replaced} 个 LLM Linear 层")


# ============================================================================
# Step 2: 修改模型结构（block 23 → block 24 权重复用）
# ============================================================================

def apply_weight_sharing(model: nn.Module):
    """让 block 24 共享 block 23 的 MSA/MLP 权重，并增加变换矩阵。

    直接在模型实例上修改 block 24 的 attn 和 mlp 引用。
    变换矩阵初始化：F1,F2 = 单位矩阵（恒等变换起点），dwconv = dirac 初始化。
    """
    blocks = model.vision_model.blocks
    src_block = blocks[SOURCE_BLOCK_IDX]
    tgt_block = blocks[TARGET_BLOCK_IDX]

    log(
        f"block {SOURCE_BLOCK_IDX} (源) → "
        f"block {TARGET_BLOCK_IDX} (目标)"
    )

    # ── 记录原始参数 id 用于验证 ────────────────────────
    tgt_attn_qkv_id_before = id(tgt_block.attn.qkv.weight)
    tgt_mlp_fc1_id_before = id(tgt_block.mlp.linear_fc1.weight)

    # ── 共享 MSA 权重 ─────────────────────────────────────
    tgt_block.attn = src_block.attn

    # ── 共享 MLP 权重 ─────────────────────────────────────
    tgt_block.mlp = src_block.mlp

    # ── LayerNorm 保持独立（不做任何修改）────────────────

    # ── 添加 Attention 变换矩阵 F1, F2 ∈ R^(M×M) ────────
    M = NUM_ATTN_HEADS  # 16
    # 初始化为单位矩阵（恒等变换起点，蒸馏后学习最优变换）
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

    # ── 添加 MLP 深度卷积变换 ────────────────────────────
    hidden_size = src_block.mlp.linear_fc1.in_features  # 1152
    kernel_size = 3
    dwconv = nn.Conv1d(
        hidden_size, hidden_size, kernel_size,
        padding=kernel_size // 2, groups=hidden_size, bias=True,
    )
    # 初始化为接近恒等映射（dirac delta）
    nn.init.dirac_(dwconv.weight)
    tgt_block.register_module("mlp_dwconv", dwconv)

    # ── 添加 MLP 变换归一化层 ───────────────────────────
    transform_norm = nn.LayerNorm(hidden_size, eps=1e-6)
    tgt_block.register_module("mlp_transform_norm", transform_norm)

    # ── 验证共享是否成功 ─────────────────────────────────
    assert tgt_block.attn is src_block.attn, "MSA 共享失败"
    assert tgt_block.mlp is src_block.mlp, "MLP 共享失败"
    assert tgt_block.norm1 is not src_block.norm1, "norm1 不应共享"
    assert tgt_block.norm2 is not src_block.norm2, "norm2 不应共享"

    log(
        f"MSA 共享验证: "
        f"{id(tgt_block.attn.qkv.weight)} == "
        f"{id(src_block.attn.qkv.weight)}"
    )
    log(
        f"MLP 共享验证: "
        f"{id(tgt_block.mlp.linear_fc1.weight)} == "
        f"{id(src_block.mlp.linear_fc1.weight)}"
    )
    log(
        f"变换矩阵: F1 {list(tgt_block.attn_transform_F1_weight.shape)}, "
        f"F2 {list(tgt_block.attn_transform_F2_weight.shape)}, "
        f"dwconv [C={hidden_size}, K={kernel_size}]"
    )

    return model


# ============================================================================
# Step 3: 生成修改后的 modeling_ising_vit.py
# ============================================================================

def generate_modified_vit_code(output_path: Path):
    """生成 MiniViT 版本的 modeling_ising_vit.py。

    在 IsingVisionEncoder.__init__ 末尾自动执行 block 23→24 的
    权重复用 + 变换矩阵注册，并修改 forward 使 block 24 走
    _forward_block_24_shared 方法。

    这样 from_pretrained 加载后自动生效，无需后处理。
    """
    src_file = Path(SOURCE_MODEL_DIR) / "modeling_ising_vit.py"
    with open(src_file) as f:
        orig_code = f.read()

    # ── 1. 在 IsingBlock.forward 后插入 _forward_block_24_shared 方法 ──
    # 找到 IsingBlock 的 forward 方法结束位置（class IsingPatchMerger 之前）
    shared_forward_method = '''

    # ★ MiniViT 权重复用：block 24 使用共享权重 + 变换矩阵
    def _forward_block_24_shared(self, x: torch.Tensor) -> torch.Tensor:
        """block 24 前向传播：共享 block 23 的 MSA/MLP 权重 + 变换矩阵。

        严格遵循 MiniViT 论文公式：
          A'_n = softmax( Σ_m F(2)_nm · Q_m K_m^T / √d )
          h_k  = Σ_n F(1)_kn · A_n · V_k
        """
        block_24 = self.blocks[24]
        block_23 = self.blocks[23]

        # ── 残差 + 共享 Attention（经变换） ────────────────
        residual = x
        x = block_24.norm1(x)

        B, N, C = x.shape
        M = block_23.attn.num_heads
        H = block_23.attn.head_dim

        qkv = block_23.attn.qkv(x).reshape(B, N, 3, M, H).permute(2, 0, 3, 1, 4)
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

        # ── 残差 + 共享 MLP（经深度卷积变换 + LayerNorm） ─
        residual = x
        x = block_24.norm2(x)

        if hasattr(block_24, "mlp_dwconv"):
            x_t = x.transpose(1, 2)
            x_t = block_24.mlp_dwconv(x_t)
            x = x_t.transpose(1, 2)

        if hasattr(block_24, "mlp_transform_norm"):
            x = block_24.mlp_transform_norm(x)

        x = block_23.mlp(x)
        x = residual + x

        return x
'''

    # 在 IsingBlock class 结束（IsingPatchMerger 开始）之前插入
    insert_before = "class IsingPatchMerger"
    if insert_before in orig_code:
        idx = orig_code.index(insert_before)
        # 找到 class IsingPatchMerger 前的最后一个空行
        orig_code = orig_code[:idx] + shared_forward_method + "\n" + orig_code[idx:]

    # ── 2. 修改 __init__：添加权重复用初始化 ──────────────
    # 在 self.merger = IsingPatchMerger(... 之后插入
    init_inject = '''
        # ── MiniViT 权重复用（自动生成）────────────────────
        _src = self.blocks[23]
        _tgt = self.blocks[24]
        _tgt.attn = _src.attn
        _tgt.mlp  = _src.mlp
        M = config.num_attention_heads
        import torch as _torch
        _tgt.register_parameter("attn_transform_F1_weight",
            _torch.nn.Parameter(_torch.eye(M), requires_grad=False))
        _tgt.register_parameter("attn_transform_F2_weight",
            _torch.nn.Parameter(_torch.eye(M), requires_grad=False))
        _dwconv = _torch.nn.Conv1d(H, H, 3, padding=1, groups=H)
        _torch.nn.init.dirac_(_dwconv.weight)
        _tgt.add_module("mlp_dwconv", _dwconv)
        _tgt.add_module("mlp_transform_norm",
            _torch.nn.LayerNorm(H, eps=config.layer_norm_eps))
'''

    insert_marker = "self.merger = IsingPatchMerger("
    if insert_marker in orig_code:
        idx = orig_code.index(insert_marker)
        line_end = orig_code.index("\n", idx)
        # 找到这行之后的下一个非空行（通常紧跟着就是）
        next_line_start = line_end + 1
        orig_code = (
            orig_code[:next_line_start]
            + init_inject
            + orig_code[next_line_start:]
        )

    # ── 3. 修改 forward：block 24 走共享路径 ────────────
    # 找到 "x = block(x)" 行，当 i==24 时替换为 _forward_block_24_shared
    # 更稳健的做法：在 forward 的 for 循环中嵌入条件判断
    old_forward_loop = "            if self.use_concat_penultimate and i == n_blocks - 2:\n                x_penultimate = x\n            x = block(x)"
    new_forward_loop = "            if self.use_concat_penultimate and i == n_blocks - 2:\n                x_penultimate = x\n            if i == 24:\n                x = self._forward_block_24_shared(x)\n            else:\n                x = block(x)"

    if old_forward_loop in orig_code:
        orig_code = orig_code.replace(old_forward_loop, new_forward_loop)
    else:
        # 备选：查找更简单的模式
        alt_pattern = "            if self.use_concat_penultimate and (i == n_blocks - 2):\n                x_penultimate = x\n            x = block(x)"
        if alt_pattern in orig_code:
            orig_code = orig_code.replace(alt_pattern, new_forward_loop)

    output_path.write_text(orig_code, encoding="utf-8")
    log(f"已生成修改后的 modeling_ising_vit.py → {output_path}")


# ============================================================================
# Step 4: 保存 MiniViT 模型权重
# ============================================================================

def save_minivit_model(model: nn.Module, output_dir: Path):
    """直接复制源模型的 safetensors 文件，仅修改 ViT block 23/24 权重。

    ★ 关键：不能通过 model.state_dict() 保存！
       AutoModel.from_pretrained 加载 AQLM 模型时会将 codebooks/codes/scales
       反量化为完整权重矩阵（~4.47B），导致保存时丢失量化压缩。
       正确做法：从源文件直接读取所有张量 → 修改 ViT 相关键 → 写入新文件。
       这样 LLM 的 AQLM 量化格式（.codebooks/.codes/.scales）得以完整保留。
    """
    log(f"保存 MiniViT 模型到 {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = Path(SOURCE_MODEL_DIR)

    # ── 1. 从源 safetensors 文件直接读取所有张量 ────────
    # 这样保留了 LLM 的 AQLM 量化格式（codebooks/codes/scales）
    all_tensors = {}
    src_index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    src_shards = sorted(set(src_index["weight_map"].values()))
    log(f"  读取源模型 {len(src_shards)} 个分片...")
    for shard_name in src_shards:
        shard_path = src_dir / shard_name
        if not shard_path.exists():
            log(f"    ⚠️ 跳过不存在的分片: {shard_path}")
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
    log(f"  从源文件读取 {len(all_tensors)} 个张量键")

    # ── 2. 移除 block 24 的冗余 MSA/MLP 权重 ──────────────
    # MiniViT 让 block 24 共享 block 23 的 attn/mlp，
    # 所以 block 24 的 MSA(qkv/proj) 和 MLP(fc1/fc2) 权重不再需要存储。
    # 仍保留 block 24 的 norm1/norm2（独立参数）。
    prefix_24 = f"vision_model.blocks.{TARGET_BLOCK_IDX}."
    keys_to_remove = [
        k for k in all_tensors
        if k.startswith(prefix_24)
        and (".attn.qkv." in k or ".attn.proj." in k or ".mlp." in k)
    ]
    for key in keys_to_remove:
        del all_tensors[key]
    log(f"  移除 block {TARGET_BLOCK_IDX} 的 {len(keys_to_remove)} 个冗余权重键")

    # ── 3. 添加 MiniViT 变换矩阵 ─────────────────────────
    tgt_block = model.vision_model.blocks[TARGET_BLOCK_IDX]

    # F1, F2 变换矩阵
    for param_name in ["attn_transform_F1_weight", "attn_transform_F2_weight"]:
        tensor = getattr(tgt_block, param_name).data.cpu()
        all_tensors[f"{prefix_24}{param_name}"] = tensor

    # MLP 深度卷积
    all_tensors[f"{prefix_24}mlp_dwconv.weight"] = \
        tgt_block.mlp_dwconv.weight.data.cpu()
    all_tensors[f"{prefix_24}mlp_dwconv.bias"] = \
        tgt_block.mlp_dwconv.bias.data.cpu()

    # Transform 归一化
    all_tensors[f"{prefix_24}mlp_transform_norm.weight"] = \
        tgt_block.mlp_transform_norm.weight.data.cpu()
    all_tensors[f"{prefix_24}mlp_transform_norm.bias"] = \
        tgt_block.mlp_transform_norm.bias.data.cpu()
    log(f"  添加 6 个变换矩阵参数（F1, F2, dwconv×2, transform_norm×2）")

    # ── 4. 写入新 safetensors 分片 ───────────────────────
    # ★ 注意：block 24 的 attn.*/mlp.* 键已从 all_tensors 中移除，
    #   不会出现在输出文件中。模型加载时由 modeling_ising_vit.py 的
    #   __init__ 通过 _tgt.attn = _src.attn 完成引用重定向，
    #   load_state_dict(strict=False) 会忽略这些缺失的键。
    max_shard_size = 2 * 1024 * 1024 * 1024  # 2GB per shard
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

    # 重命名分片（带总数）
    total_shards = shard_idx
    for i in range(total_shards):
        old = output_dir / f"model-{i + 1:05d}-of-00000.safetensors"
        new = output_dir / f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
            for k, v in list(weight_map.items()):
                if v == f"model-{i + 1:05d}-of-00000.safetensors":
                    weight_map[k] = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"

    # 保存索引
    index_data = {"metadata": {}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)
    log(f"  已保存 {total_shards} 个权重分片，{len(weight_map)} 个键")

    # ── 5. 复制配置/分词器文件 ──────────────────────────
    files_to_copy = [
        "config.json", "generation_config.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json",
        "configuration_riverone_qc.py", "modeling_riverone_qc.py",
        "conversation.py",
        "preprocessor_config.json", "processor_config.json",
        "chat_template.jinja", "video_preprocessor_config.json",
    ]

    for filename in files_to_copy:
        src = src_dir / filename
        dst = output_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # ── 6. 生成修改后的 modeling_ising_vit.py ───────────
    generate_modified_vit_code(output_dir / "modeling_ising_vit.py")

    # ── 7. 复制量化配置（保留 AQLM LLM 量化信息）───────
    quant_src = src_dir / "quant_config.json"
    if quant_src.exists():
        shutil.copy2(str(quant_src), str(output_dir / "quant_config.json"))

    # ── 8. 保存 MiniViT 压缩记录 ────────────────────────
    minivit_config = {
        "compression_method": "MiniViT",
        "source_block": SOURCE_BLOCK_IDX,
        "target_block": TARGET_BLOCK_IDX,
        "shared_components": ["MSA (qkv + proj)", "MLP (fc1 + fc2)"],
        "independent_components": ["norm1", "norm2"],
        "added_params": {
            "attn_transform_F1": [NUM_ATTN_HEADS, NUM_ATTN_HEADS],
            "attn_transform_F2": [NUM_ATTN_HEADS, NUM_ATTN_HEADS],
            "mlp_dwconv": "Conv1d(C=1152, K=3, groups=1152)",
            "mlp_transform_norm": "LayerNorm(1152)",
        },
        "source_model": SOURCE_MODEL_DIR,
    }
    with open(output_dir / "miniViT_config.json", "w") as f:
        json.dump(minivit_config, f, indent=2)

    log(f"  MiniViT 模型已完整保存到: {output_dir}")
    log(f"  ★ AQLM 量化格式已保留，LLM 权重仍为压缩态")


# ============================================================================
# 主入口
# ============================================================================

def main():
    log("=" * 60)
    log(" RiverOne-QC-4B-v1 MiniViT 权重复用压缩")
    log(f" 源模型: {SOURCE_MODEL_DIR}")
    log(f" 输出目录: {OUTPUT_DIR}")
    log(f" 压缩范围: block {SOURCE_BLOCK_IDX} → {TARGET_BLOCK_IDX}")
    log("=" * 60)

    # 1. 加载源模型
    model = load_source_model()

    # 2. 应用权重复用
    model = apply_weight_sharing(model)

    # 3. 保存
    save_minivit_model(model, OUTPUT_DIR)

    log("=" * 60)
    log(" 完成！执行以下命令进行下一步：")
    log(f"  1. 验证: python3 {SCRIPT_DIR / 'verify_minivit.py'}")
    log(f"  2. 蒸馏: python3 {SCRIPT_DIR / 'distill_minivit.py'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
