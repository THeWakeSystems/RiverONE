#!/usr/bin/env python3
"""
=============================================================================
 RiverOne-QC-4B-v1 模型定向 AQLM 量化主脚本（全部36层 LLM）
=============================================================================
 脚本功能：
   - 加载 RiverOne-QC-4B-v1 源模型，自动解析模型结构，定位语言模型(LLM)分支
   - 对 LLM 全部 36 层 Transformer 的 Attention + MLP 线性权重执行 AQLM 量化
   - 量化方案：AQLM 1×16 scheme（in_group_size=16, out_group_size=1,
     num_codebooks=1, nbits_per_codebook=16）
   - 量化完成后保存完整模型权重与配置到输出目录
   - 全程输出量化进度日志到 quantize.log

 适用场景：
   - 在 Linux 服务器上对 RiverOne-QC-4B-v1（miniViT 版本）多模态模型的
     LLM 分支进行全部36层定向量化
   - 保留视觉编码器（miniViT）、投影层、Embedding/LM Head 等组件的原始精度

 运行环境：
   - Python 3.10+, CUDA 12.x, PyTorch 2.x, transformers 4.x
   - 已安装 aqlm（PyPI 官方包）及 AQLM 量化引擎（来自 Vahe1994/AQLM）

 与 RiverOne-QC-4B-AQLM-L 的区别：
   - 源模型为 RiverOne-QC-4B-v1（使用 miniViT 视觉编码器）
   - 量化全部 36 层 LLM（而非仅最后 N 层）
   - 输出目录位于 RiverOne-QC-4B-v1-AQLM-miniViT/

 核心参数概览（详见脚本顶部配置区）：
   - SOURCE_MODEL_PATH : 源模型路径
   - OUTPUT_DIR        : 量化输出目录
   - NUM_LAST_LAYERS   : 量化的最后 N 层（36 = 全部层）
   - IN_GROUP_SIZE     : AQLM 输入组大小（1×16 方案中为 16）
   - OUT_GROUP_SIZE    : AQLM 输出组大小（1×16 方案中为 1）
   - NUM_CODEBOOKS     : 码本数量（1×16 方案中为 1）
   - NBITS_PER_CODEBOOK: 每个码本的比特数（16）
   - NSAMPLES          : 校准数据样本数
   - MODEL_SEQLEN      : 校准序列长度
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import gc
import json
import logging
import random
import math
import copy
from argparse import Namespace
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 将本地 AQLM 库路径加入 sys.path
# engine/ 包含从 AQLM 官方仓库复制的核心量化模块，确保项目自包含
# 注意：engine 位于 quantize 的上级目录（项目根目录）
# ---------------------------------------------------------------------------
_AQLM_LIB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "engine"
)
if _AQLM_LIB_PATH not in sys.path:
    sys.path.insert(0, _AQLM_LIB_PATH)

# ---------------------------------------------------------------------------
# 导入 AQLM 量化引擎核心模块（来自本地 engine/）
# ---------------------------------------------------------------------------
from aq_engine import AQEngine                     # 单层量化引擎（收集 Hessian + 执行量化）
from src.aq import QuantizedWeight, QuantizedLinear  # 量化权重表示 & 训练用 QuantizedLinear
from src.modelutils import (
    get_model,
    get_layers,
    get_llm_model,
    get_llm_config,
    get_forward_model,
    get_hidden_size,
    get_use_cache,
    set_use_cache,
    get_quantizer_key_prefix,
    find_sublayers,
    get_sequential_groups,
    is_internvl,
)
from src.datautils import get_wikitext2, set_seed
from src.utils import using_tf32

# AQLM 官方 init_aq_engines（wrapper 模式收集激活值）
# 同时导入并行版本，支持多 GPU 分布式量化
from main import init_aq_engines, init_aq_engines_parallel, update_outs_parallel

# AQLM 推理库（PyPI 官方 aqlm 包），用于最终保存时的格式兼容
from aqlm import QuantizedLinear as AQLMInferenceQuantizedLinear
from aqlm.utils import _dequantize_weight, unpack_int_data, get_int_dtype

# ---------------------------------------------------------------------------
# 补丁：让 AQLM 工具库识别 RiverOne-QC 混合模型架构
# src/modelutils.py 的 INTERNVL_TYPES 仅包含 "internvl_chat"，
# 需要添加 "riverone_qc" 使其正确识别 LLM 子配置（llm_config）
# ---------------------------------------------------------------------------
import src.modelutils as _mu
if "riverone_qc" not in _mu.INTERNVL_TYPES:
    _mu.INTERNVL_TYPES = tuple(list(_mu.INTERNVL_TYPES) + ["riverone_qc"])
    print("[补丁] 已将 'riverone_qc' 注册到 INTERNVL_TYPES，确保 LLM 配置正确解析")

# 忽略 transformers 中的警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 配置参数区 —— 所有可调参数集中于此，行尾注释说明含义
# ============================================================================

# --- 路径配置 ---
# 源模型绝对路径（RiverOne-QC-4B-v2，使用 miniViT 视觉编码器）
SOURCE_MODEL_PATH: str = "/home/lxy/workspace/riverone-release/RiverOne-QC-4B-v2"
# 量化输出目录（全部36层）
OUTPUT_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "RiverOne-QC-4B-v2-AQLM-36L"
)
# 量化日志文件路径（保存到 scripts 目录）
LOG_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "quantize.log"
)

# --- 量化范围配置 ---
NUM_LAST_LAYERS: int = 36         # 量化全部 36 层 LLM
LLM_KEYWORDS: List[str] = [       # 用于自动定位 LLM 分支的属性名关键词
    "language_model", "llm", "decoder",
]

# --- AQLM 1×16 方案参数（硬性约束，按 AQLM 源码约定）---
# AQLM 中 "1×16" 含义：每 1 个输出通道 × 16 个输入通道为一组
# kernel_selector 匹配条件：out_group_size=1, in_group_size in [8,16]
IN_GROUP_SIZE: int = 16           # 输入维度分组大小（1×16 方案固定为 16）
OUT_GROUP_SIZE: int = 1           # 输出维度分组大小（1×16 方案固定为 1）
NUM_CODEBOOKS: int = 1            # 码本数量（1×16 方案固定为 1）
NBITS_PER_CODEBOOK: int = 16      # 每个码本的比特数（16 → codebook_size=65536）
ATTENTION_NBITS_PER_CODEBOOK: Optional[int] = None  # attention 层专用 nbits（None=与 NBITS_PER_CODEBOOK 一致）
                                                    # 设为 8 则 attention 的 q/k/v/o 用 codebook_size=256，
                                                    # 解决 attention 小层 codebook 开销过大问题
SUBlayer_NBITS: dict = {}            # 细粒度 sublayer nbits 覆盖，如 {"k_proj": 14, "v_proj": 14}

# --- 量化训练超参数 ---
CODEBOOK_VALUE_NBITS: int = 16    # 码本值的比特数（16=无损，<16 有损压缩码本）
CODEBOOK_VALUE_NUM_GROUPS: int = 1  # 码本值分组数
SCALE_NBITS: int = 0              # 量化 scales 的比特数（0=不量化scales，保持浮点精度）
INIT_MAX_ITER: int = 20           # K-Means 初始化最大迭代次数
INIT_MAX_POINTS_PER_CENTROID: int = 2  # K-Means 每个聚类中心的最大采样点数
                                       # 1×16 方案 codebook=65536，设为 2 则最多采样 131072 点
                                       # 原默认 1000000 会导致全量数据聚类，OOM
USE_FAISS: bool = False           # 是否使用 FAISS K-Means（k-means++ 初始化 + GPU 加速）
                                  # 设为 True 显著提升 65536 中心大 codebook 的聚类质量
FAISS_NREDO: int = 1             # FAISS K-Means 重复运行次数（取最优）。>1 可降低局部最优风险，但耗时为 nredo 倍
MAX_EPOCHS: int = 5               # 量化优化最大 epoch 数
STEPS_PER_EPOCH: int = 50         # 每个 epoch 的优化步数
LR: float = 1e-4                  # 学习率（Adam）
BEAM_SIZE: int = 1                # Beam search 大小
RELATIVE_MSE_TOLERANCE: float = None  # 早停 MSE 相对容忍度（None=不使用早停）

# --- 校准数据配置 ---
NSAMPLES: int = 64                # 校准数据样本数（越大量化越精确，但越占显存）
MODEL_SEQLEN: int = 2048          # 每个样本的序列长度
DATASET: str = "wikitext2"        # 校准数据集名称（wikitext2 / c4 / red_pajama）
SEED: int = 42                    # 随机种子，确保可复现
VAL_SIZE: int = 0                 # 验证集大小（0=不从校准集中切分验证集）

# --- 微调（finetune）配置 ---
FINETUNE_MAX_EPOCHS: int = 0      # 量化后微调 epoch 数（0=跳过微调阶段）
FINETUNE_LR: float = 1e-5         # 微调学习率
FINETUNE_BATCH_SIZE: int = 1      # 微调批次大小

# --- 硬件/性能配置 ---
DEVICES: List[str] = ["cuda:0"]   # 层传播/量化训练使用的 GPU
# k-means 聚类可用的所有 GPU（用于 find_nearest_cluster 分块并行，缓解显存瓶颈）
# 注意：仅使用 GPU6 进行 k-means，避免跨卡通信开销
KMEANS_DEVICES: List[str] = ["cuda:0"]
DTYPE: str = "bfloat16"           # 模型加载精度（bfloat16 推荐，节省显存）
ATTN_IMPLEMENTATION: str = "eager"  # 注意力实现方式（eager/flash_attention_2）
OFFLOAD_ACTIVATIONS: bool = True  # 激活值卸载到 CPU，节省 GPU 显存
USE_CHECKPOINTING: bool = False   # 是否对量化层启用梯度检查点
SKIP_OUT_LOSS: bool = False       # 是否跳过输出 loss 计算（加速量化）
TRUE_SEQUENTIAL: bool = True      # 是否逐子层顺序量化（推荐开启，提升精度）
PRINT_FREQUENCY: int = 10         # 日志打印频率（每 N 步打印一次 loss）
RESUME: bool = False              # 是否从断点恢复

# --- 线性层筛选关键词 ---
# 仅名字匹配这些后缀的 nn.Linear 层会被量化
LINEAR_LAYER_KEYWORDS: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention 线性投影
    "gate_proj", "up_proj", "down_proj",       # MLP 线性层
]


# ============================================================================
# 日志系统初始化
# ============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    """初始化双通道日志系统：同时输出到控制台和日志文件。

    Args:
        log_file: 日志文件绝对路径。

    Returns:
        配置完成的 Logger 实例。
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("aqlm_quantize")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 控制台 handler（INFO 级别）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)

    # 文件 handler（DEBUG 级别，完整日志）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging(LOG_FILE)


# ============================================================================
# 模型结构解析与 LLM 定位
# ============================================================================

def locate_language_model(model: nn.Module) -> nn.Module:
    """自动定位模型中的语言模型(LLM)分支。

    通过遍历模型的直接子属性，按关键词列表匹配定位 LLM 组件。
    支持 InternVL 系列（language_model）、纯 LLM、自定义混合架构。

    Args:
        model: 完整的预训练模型实例。

    Returns:
        定位到的语言模型子模块。

    Raises:
        AttributeError: 当无法定位到 LLM 分支时抛出。
    """
    # 优先检测常见属性名
    for attr_name in LLM_KEYWORDS:
        if hasattr(model, attr_name):
            llm = getattr(model, attr_name)
            logger.info(
                f"[模型解析] 通过属性 '{attr_name}' 定位到 LLM 分支: "
                f"{type(llm).__name__}"
            )
            return llm

    # 遍历直接子属性做兜底匹配
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(model, attr_name)
            if isinstance(attr, nn.Module):
                for keyword in LLM_KEYWORDS:
                    if keyword in attr_name.lower():
                        logger.info(
                            f"[模型解析] 通过关键词 '{keyword}' 匹配属性 "
                            f"'{attr_name}' -> {type(attr).__name__}"
                        )
                        return attr
        except Exception:
            continue

    raise AttributeError(
        f"[错误] 无法定位 LLM 分支！尝试的属性名: {LLM_KEYWORDS}。"
        f"请检查模型结构或手动扩展 LLM_KEYWORDS 列表。"
    )


def get_llm_layers(model: nn.Module) -> nn.ModuleList:
    """获取 LLM 分支的 Transformer 层列表。

    从 LLM 模块中提取 decoder layers，支持 Qwen3/LLaMA/InternVL 等架构。

    Args:
        model: 完整的预训练模型实例。

    Returns:
        包含所有 Transformer 层的 ModuleList。

    Raises:
        ValueError: 当无法找到层列表时抛出。
    """
    return get_layers(model)


def resolve_target_layers(
    model: nn.Module,
    num_last_layers: int,
) -> Tuple[nn.Module, nn.ModuleList, List[int], Dict[int, str]]:
    """解析模型结构，定位 LLM 分支和待量化的最后 N 层。

    执行流程：
    1. 定位 LLM 分支（language_model 属性）
    2. 获取全部 Transformer 层列表
    3. 计算最后 N 层的索引范围
    4. 获取各子层名称，建立索引→名称映射

    Args:
        model: 完整模型实例。
        num_last_layers: 要量化的最后 N 层数量（36 = 全部层）。

    Returns:
        (llm_module, all_layers, target_indices, index_to_name)
        - llm_module: LLM 分支模块
        - all_layers: 全部 Transformer 层 ModuleList
        - target_indices: 目标层索引列表（如 [0,1,...,35]）
        - index_to_name: 层索引→完整模块路径映射
    """
    logger.info("=" * 60)
    logger.info("[模型解析] 开始解析模型结构...")

    # Step 1: 定位 LLM 分支
    llm = locate_language_model(model)
    logger.info(f"[模型解析] LLM 分支类型: {type(llm).__name__}")
    llm_config = get_llm_config(model)
    logger.info(
        f"[模型解析] LLM 配置: hidden_size={llm_config.hidden_size}, "
        f"num_hidden_layers={llm_config.num_hidden_layers}"
    )

    # Step 2: 获取全部层
    all_layers = get_llm_layers(model)
    total_layers = len(all_layers)
    logger.info(f"[模型解析] Transformer 总层数: {total_layers}")

    # Step 3: 确认最后 N 层的索引
    if num_last_layers > total_layers:
        raise ValueError(
            f"[错误] 请求量化最后 {num_last_layers} 层，但模型只有 {total_layers} 层！"
        )
    target_indices = list(range(total_layers - num_last_layers, total_layers))
    logger.info(
        f"[模型解析] 目标量化层索引: {target_indices} "
        f"(最后 {num_last_layers} 层，层号 "
        f"{target_indices[0]}~{target_indices[-1]})"
    )

    # Step 4: 获取量化器前缀（用于构建完整权重名）
    key_prefix = get_quantizer_key_prefix(model)
    logger.info(f"[模型解析] 量化器键前缀: '{key_prefix}'")

    # Step 5: 构建索引→名称映射，并输出层级清单
    logger.info("-" * 60)
    logger.info("[层级匹配清单] 以下为待量化的最后 N 层及其子模块:")
    index_to_name = {}
    for idx in target_indices:
        layer = all_layers[idx]
        sublayer_names = list(find_sublayers(layer).keys())
        # 筛选出需要量化的线性层
        quantizable = [
            n for n in sublayer_names
            if any(kw in n for kw in LINEAR_LAYER_KEYWORDS)
        ]
        layer_path = f"{key_prefix}.{idx}"
        index_to_name[idx] = layer_path
        logger.info(
            f"  层 {idx:2d} ({layer_path}): 共 {len(sublayer_names)} 个子层, "
            f"可量化线性层 {len(quantizable)} 个 -> {quantizable}"
        )

    # Step 6: 输出非量化组件清单（确认不会误量化）
    logger.info("-" * 60)
    logger.info("[排除清单] 以下组件保持原始精度，不会被量化:")
    excluded_components = []
    for name, _ in model.named_parameters():
        # 跳过目标层中的可量化线性层
        is_target = any(
            f"{key_prefix}.{idx}" in name for idx in target_indices
        )
        is_quantizable_weight = any(kw in name for kw in LINEAR_LAYER_KEYWORDS)
        if is_target and is_quantizable_weight:
            continue
        if "weight" in name:  # 只列出权重参数（不含 bias/norm）
            excluded_components.append(name)
    for comp in excluded_components[:20]:  # 前 20 个样例
        logger.info(f"  [保留] {comp}")
    if len(excluded_components) > 20:
        logger.info(
            f"  ... 及另外 {len(excluded_components) - 20} 个参数"
        )

    logger.info("=" * 60)
    return llm, all_layers, target_indices, index_to_name


# ============================================================================
# 辅助函数：Rotary Position Embeddings
# ============================================================================

def get_rotary_emb_module(model: nn.Module):
    """从 LLM 模型中获取 rotary embedding 模块。

    新版 transformers（>=4.57）的 Qwen3 要求在调用 decoder layer 时
    显式传入 position_embeddings=(cos, sin)，否则 attention 层报错：
    "TypeError: cannot unpack non-iterable NoneType object"

    Args:
        model: 完整的预训练模型。

    Returns:
        rotary_emb 模块。
    """
    llm = get_llm_model(model)
    if hasattr(llm, "model") and hasattr(llm.model, "rotary_emb"):
        return llm.model.rotary_emb
    raise AttributeError(
        "无法定位 rotary_emb 模块，请检查 LLM 架构。"
    )


def make_position_embeddings(
    rotary_emb: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 (cos, sin) rotary position embeddings。

    Args:
        rotary_emb: rotary embedding 模块。
        hidden_states: [batch, seq_len, hidden_size]。
        position_ids: [batch, seq_len]。

    Returns:
        (cos, sin) 元组。
    """
    cos, sin = rotary_emb(hidden_states, position_ids)
    return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


# ============================================================================
# 校准数据准备
# ============================================================================

def prepare_calibration_data(
    model: nn.Module,
    tokenizer,
    nsamples: int,
    seqlen: int,
    dataset_name: str,
    seed: int,
    devices: List[torch.device],
    offload_activations: bool,
) -> Tuple[List[torch.Tensor], Dict]:
    """准备 AQLM 量化所需的校准数据（层输入激活值）。

    通过 mock 前向传播收集每一层的第一层输入激活值。

    Args:
        model: 完整模型实例。
        tokenizer: 与模型配套的分词器。
        nsamples: 校准样本数量。
        seqlen: 序列截断长度。
        dataset_name: 校准数据集名称。
        seed: 随机种子。
        devices: GPU 设备列表。
        offload_activations: 是否将激活值卸载到 CPU。

    Returns:
        (calib_data, forward_args): 校准激活值列表和前向参数。
    """
    logger.info("[校准数据] 加载校准数据集...")
    set_seed(seed)

    # 根据配置选择数据集
    if dataset_name == "wikitext2":
        data = get_wikitext2(nsamples, seqlen, tokenizer)
    elif dataset_name == "c4":
        from src.datautils import get_c4
        data = get_c4(nsamples, seqlen, tokenizer)
    elif dataset_name == "red_pajama":
        from src.datautils import get_red_pajama
        data = get_red_pajama(nsamples, seqlen, tokenizer)
    else:
        raise ValueError(
            f"[错误] 不支持的数据集: {dataset_name}，"
            f"可选: wikitext2, c4, red_pajama"
        )

    logger.info(
        f"[校准数据] 数据集: {dataset_name}, 样本数: {len(data)}, "
        f"序列长度: {seqlen}"
    )

    # 使用手动实现的 collect_layer_inputs 收集层输入
    logger.info(
        "[校准数据] 开始收集各层输入激活值（可能耗时较长）..."
    )
    inps, forward_args = collect_layer_inputs(
        model, data, seqlen, devices, offload_activations
    )
    logger.info(
        f"[校准数据] 激活值收集完成，"
        f"共 {sum(t.shape[0] for t in inps)} 个样本"
    )
    return inps, forward_args


def collect_layer_inputs(
    model: nn.Module,
    data: Sequence[torch.Tensor],
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[List[torch.Tensor], Dict]:
    """收集 LLM 第一层的输入激活值（简化版 get_inps）。

    该函数是 AQLM 官方 get_inps 的简化适配版，专为 RiverOne-QC 混合模型调整。
    核心思路：对纯文本 forward 路径进行 mock，捕获 LLM 第一层的输入。

    Args:
        model: 完整模型。
        data: 校准数据 token 序列列表。
        model_seqlen: 序列长度。
        devices: GPU 设备列表。
        offload_activations: 是否卸载到 CPU。

    Returns:
        (inps, forward_args): 层输入和 forward 参数。
    """
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")

    # 处理数据格式：将长序列拆分为多个短序列
    if isinstance(data, torch.Tensor) and data.shape[0] == 1:
        num_sequences = data.numel() // model_seqlen
        data = [
            data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device)
            for i in range(num_sequences)
        ]
        logger.info(f"[数据] 拆分长序列为 {len(data)} 个短序列")

    # 获取 embedding 层
    llm = get_llm_model(model)
    emb = llm.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)

    device = emb.weight.device
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    hidden_size = get_hidden_size(model)

    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (
                min(nsamples_per_device, len(data) - i * nsamples_per_device),
                model_seqlen,
                hidden_size,
            ),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]

    forward_model = get_forward_model(model)
    cache = {"i": 0}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        """钩子模块：捕获第一层的输入激活值后提前退出。"""
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__("module")
            return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][
                cache["i"] % nsamples_per_device
            ] = inp
            cache["i"] += 1
            raise CatcherExit()

    layers[0] = Catcher(layers[0])

    # 逐批前向，捕获输入
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            forward_model(
                batch_inps,
                attention_mask=torch.ones_like(batch_inps),
            )
        except CatcherExit:
            pass  # 正常退出：Catcher 已记录激活值

    # 恢复原始层
    layers[0] = layers[0].module
    layers[0] = layers[0].to(layer_device)
    if emb_device.type != "cuda":
        emb = emb.to(emb_device)
    torch.cuda.empty_cache()

    # 构建 forward_args：包含 position_embeddings 所需信息
    # 新版 transformers Qwen3 要求在调用 decoder layer 时显式传入
    # position_embeddings=(cos, sin)，否则 attention 层会报错
    rotary_emb = get_rotary_emb_module(model)
    # 默认 position_ids：[0, 1, ..., seqlen-1]，对所有样本相同
    default_position_ids = torch.arange(
        model_seqlen, device=device
    ).unsqueeze(0)

    forward_args = {
        "rotary_emb": rotary_emb,
        "default_position_ids": default_position_ids,
    }
    logger.info(f"[数据收集] 共捕获 {cache['i']} 个激活值张量")
    return inps, forward_args


# ============================================================================
# AQLM 量化执行核心
# ============================================================================

def quantize_single_layer(
    layer: nn.Module,
    layer_idx: int,
    inps: List[torch.Tensor],
    outs: List[torch.Tensor],
    args: Namespace,
    forward_args: Dict,
    model: nn.Module,  # 完整模型引用，用于 get_sequential_groups
) -> nn.Module:
    """对单个 Transformer 层内所有目标线性子层执行 AQLM 量化。

    该函数是量化操作的核心执行单元：
    1. 将该层移动到 GPU
    2. 识别层内所有需要量化的 nn.Linear 子层
    3. 为每个目标子层创建 AQEngine，收集 Hessian 信息
    4. 执行 AQLM 量化训练（K-Means 初始化 + 梯度优化 + Beam Search）
    5. 将原始 nn.Linear 替换为量化后的 QuantizedLinear

    Args:
        layer: 待量化的 Transformer 层（DecoderLayer）。
        layer_idx: 层索引（全局编号）。
        inps: 该层的输入激活值。
        outs: 该层的输出激活值缓冲区。
        args: 量化参数命名空间。
        forward_args: forward 额外参数（rotary_emb 等）。
        model: 完整模型引用（用于 get_sequential_groups）。

    Returns:
        量化后的层（原始 nn.Linear 已被替换为 QuantizedLinear）。
    """
    logger.info(f"[量化] 开始处理第 {layer_idx} 层...")

    # 保存原始设备/精度信息，后续恢复
    layer_device_original = next(layer.parameters()).device
    layer_dtype_original = next(layer.parameters()).dtype
    num_devices = len(args.devices)

    # 将层移到主 GPU 并转为 float32
    # AQLM 的 QuantizedLinear 内部权重始终为 float32，
    # 必须保证层 dtype 与之匹配，否则 F.linear 会报 dtype mismatch
    layer = layer.to(device=args.devices[0], dtype=torch.float32)

    # 同步将输入/输出激活值转为 float32，并分发到各 GPU
    inps_float32 = [
        inp.to(
            device=args.devices[min(i, num_devices - 1)],
            dtype=torch.float32,
        )
        for i, inp in enumerate(inps)
    ]
    outs_float32 = [
        out.to(
            device=args.devices[min(i, num_devices - 1)],
            dtype=torch.float32,
        )
        for i, out in enumerate(outs)
    ]

    # 获取子层列表（按依赖顺序排列，true_sequential 模式下逐子层量化）
    # 注意：get_sequential_groups 需要完整模型来判断 LLM 架构类型
    if args.true_sequential:
        sequential_groups = get_sequential_groups(model)
    else:
        sequential_groups = [list(find_sublayers(layer).keys())]

    for group_idx, sublayer_names in enumerate(sequential_groups):
        # 筛选需要量化的子层（仅线性层，排除 norm/embedding 等）
        quantizable_names = [
            name for name in sublayer_names
            if any(kw in name for kw in LINEAR_LAYER_KEYWORDS)
        ]
        if not quantizable_names:
            logger.debug(
                f"  子层组 {group_idx}: 无可量化线性层，跳过"
            )
            continue

        logger.info(
            f"  子层组 {group_idx}: 待量化 {len(quantizable_names)} 个子层 "
            f"-> {quantizable_names}"
        )

        # 预计算 position_embeddings（新版 Qwen3 必需）
        engine_forward_args = {}
        rotary_emb_fwd = forward_args.get("rotary_emb", None)
        default_pos_ids = forward_args.get("default_position_ids", None)
        if rotary_emb_fwd is not None and default_pos_ids is not None:
            sample_x = inps_float32[0][0:1]
            pos_ids = default_pos_ids[:, :sample_x.shape[1]].to(
                args.devices[0]
            )
            cos, sin = rotary_emb_fwd(sample_x, pos_ids)
            engine_forward_args["position_embeddings"] = (cos, sin)

        # init_aq_engines 在单 GPU 上运行；
        # k-means 内部通过 args.devices 自动多卡并行
        aq_engines = init_aq_engines(
            layer,
            quantizable_names,
            inps_float32[0],
            outs_float32[0],
            **engine_forward_args,
        )

        # 逐个子层执行量化
        for sublayer_name, engine in aq_engines.items():
            logger.info(
                f"    [{sublayer_name}] 开始 AQLM 量化..."
            )
            t_start = time.time()

            try:
                # ★ attention/特定 sublayer 专用 nbits
                _orig_nbits = args.nbits_per_codebook
                _use_attn_nbits = (
                    getattr(args, 'attention_nbits_per_codebook', None) is not None
                    and 'self_attn' in sublayer_name
                )
                # 细粒度 nbits: per-projection override (优先级最高)
                _sublayer_nbits = getattr(args, 'sublayer_nbits', {})
                _sublayer_nbits_override = None
                for kw, nb in _sublayer_nbits.items():
                    if kw in sublayer_name:
                        _sublayer_nbits_override = nb
                        break
                if _sublayer_nbits_override is not None:
                    args.nbits_per_codebook = _sublayer_nbits_override
                    logger.info(
                        f"    [{sublayer_name}] sublayer 专用 nbits={args.nbits_per_codebook} "
                        f"(codebook_size={2**args.nbits_per_codebook})"
                    )
                elif _use_attn_nbits:
                    args.nbits_per_codebook = args.attention_nbits_per_codebook
                    logger.info(
                        f"    [{sublayer_name}] attention 专用 nbits={args.nbits_per_codebook} "
                        f"(codebook_size={2**args.nbits_per_codebook})"
                    )

                quantized_weight = engine.quantize(
                    args=args, verbose=True
                )

                if _use_attn_nbits or _sublayer_nbits_override is not None:
                    args.nbits_per_codebook = _orig_nbits  # 恢复默认值

                # 替换原始 nn.Linear → QuantizedLinear
                with torch.no_grad():
                    new_linear = QuantizedLinear(
                        quantized_weight,
                        engine.layer.bias,
                    )
                    if args.use_checkpointing:
                        new_linear.use_checkpoint = True

                    # 在层内递归查找并替换原始层引用
                    found = False
                    for submodule in layer.modules():
                        for (
                            child_name,
                            child_module,
                        ) in submodule.named_children():
                            if child_module is engine.layer:
                                setattr(
                                    submodule,
                                    child_name,
                                    new_linear,
                                )
                                found = True
                                break
                        if found:
                            break

                    if not found:
                        logger.error(
                            f"    [{sublayer_name}] 错误："
                            f"无法在模型中找到原始层引用！"
                        )
                        raise RuntimeError(
                            f"无法定位子层 {sublayer_name} 的父模块"
                        )

                # 统计量化比特数
                weight_avg_bits = (
                    quantized_weight.estimate_nbits_per_parameter()
                )
                num_params = torch.numel(engine.layer.weight.data)
                logger.info(
                    f"    [{sublayer_name}] 量化完成！"
                    f" 平均比特/参数: {weight_avg_bits:.2f}, "
                    f" 参数量: {num_params:,}, "
                    f" 耗时: {time.time() - t_start:.1f}s"
                )

            except Exception as e:
                logger.error(
                    f"    [{sublayer_name}] 量化失败: {e}"
                )
                # 显存不足时给出明确提示
                if "out of memory" in str(e).lower():
                    logger.error(
                        "[显存不足] 请尝试: 1) 减少 NSAMPLES "
                        "2) 减小 MODEL_SEQLEN "
                        "3) 开启 OFFLOAD_ACTIVATIONS "
                        "4) 使用更大显存的 GPU"
                    )
                raise

        # 释放 AQEngine 引用，回收显存
        del aq_engines
        gc.collect()
        torch.cuda.empty_cache()

    # 将 float32 输出回写到原始 bfloat16 缓冲区（供后续层传播使用）
    for i in range(len(outs)):
        outs[i].copy_(
            outs_float32[i].to(
                device=outs[i].device, dtype=layer_dtype_original
            )
        )

    # 恢复层到原始设备和精度
    layer = layer.to(
        device=layer_device_original, dtype=layer_dtype_original
    )
    logger.info(f"[量化] 第 {layer_idx} 层处理完毕")
    return layer


# ============================================================================
# 层间激活传播（逐层 forward，获取下一层输入）
# ============================================================================

@torch.no_grad()
def update_outs(
    layer: nn.Module,
    inps: List[torch.Tensor],
    outs: List[torch.Tensor],
    **forward_args,
) -> None:
    """将当前层的输出传播为下一层的输入，更新 outs 缓冲区。

    对 inps 中每个设备的张量，逐序列执行 layer.forward()，将输出写入 outs。
    自动计算 rotary position embeddings 以兼容新版 transformers Qwen3。

    Args:
        layer: 当前 Transformer 层。
        inps: 当前层的输入激活值列表（每设备一个张量）。
        outs: 输出缓冲区（会被原地更新为当前层的输出）。
        **forward_args: 包含 rotary_emb, default_position_ids 等。
    """
    device = next(layer.parameters()).device
    layer_dtype = next(layer.parameters()).dtype  # 层参数精度（bf16/fp32）
    # rotary embedding 模块
    rotary_emb = forward_args.pop("rotary_emb", None)
    # 默认 position_ids
    default_pos_ids = forward_args.pop("default_position_ids", None)

    for i, inp_tensor in enumerate(inps):
        seq_len = inp_tensor.shape[1]
        # 为当前序列长度准备 position_ids 和 position_embeddings
        if default_pos_ids is not None and rotary_emb is not None:
            pos_ids = default_pos_ids[:, :seq_len].to(device)
        else:
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        for j in range(len(inp_tensor)):
            # 将输入转为与层参数一致的 dtype，避免 F.linear 报 dtype mismatch
            x = inp_tensor[j].to(device=device, dtype=layer_dtype).unsqueeze(0)
            # [1, seq_len, hidden_size]

            # 计算 position_embeddings（新版 Qwen3 必需）
            layer_kwargs = {}
            if rotary_emb is not None:
                cos, sin = rotary_emb(x, pos_ids)
                layer_kwargs["position_embeddings"] = (cos, sin)

            out = layer(x, **layer_kwargs)[0]
            # 写回 outs（保持原始设备和 pin_memory 状态）
            outs[i][j].copy_(
                out.reshape_as(outs[i][j]), non_blocking=True
            )

    # 恢复 forward_args（供后续使用）
    if rotary_emb is not None:
        forward_args["rotary_emb"] = rotary_emb
    if default_pos_ids is not None:
        forward_args["default_position_ids"] = default_pos_ids


# ============================================================================
# 量化后模型保存
# ============================================================================

def convert_to_inference_format(layer: nn.Module) -> nn.Module:
    """将训练用 QuantizedLinear（src.aq）转换为推理用 QuantizedLinear（aqlm.inference）。

    AQLM 仓库中有两个 QuantizedLinear：
    - src.aq.QuantizedLinear: 包含 QuantizedWeight，用于训练/量化阶段
    - aqlm.inference.QuantizedLinear: PyPI 官方推理模块，有高效的 CUDA kernel

    此函数遍历层内所有模块，将训练版转换为推理版。

    Args:
        layer: 包含训练版 QuantizedLinear 的模块。

    Returns:
        转换后的模块。
    """
    for submodule in layer.modules():
        for child_name, child_module in list(
            submodule.named_children()
        ):
            if isinstance(child_module, QuantizedLinear) and not isinstance(
                child_module, AQLMInferenceQuantizedLinear
            ):
                qw = child_module.quantized_weight
                # 获取量化参数
                in_features = qw.in_features
                out_features = qw.out_features
                in_group_size = qw.in_group_size
                out_group_size = qw.out_group_size
                num_codebooks = qw.num_codebooks
                nbits_per_codebook = qw.nbits_per_codebook

                # 创建推理版 QuantizedLinear
                # CUDA kernel code1x16_matmat 要求 input 与 codebooks 同 dtype
                # 模型精度为 bf16，码本也转 bf16（CUDA kernel 仍可加速）
                infer_linear = AQLMInferenceQuantizedLinear(
                    in_features=in_features,
                    out_features=out_features,
                    in_group_size=in_group_size,
                    out_group_size=out_group_size,
                    num_codebooks=num_codebooks,
                    nbits_per_codebook=nbits_per_codebook,
                    bias=child_module.bias is not None,
                    device=next(qw.parameters()).device,
                    dtype=torch.bfloat16,
                )

                # 复制量化参数（fp32 → bf16，匹配模型精度）
                infer_linear.codebooks.data.copy_(
                    qw.codebooks.data.to(torch.bfloat16)
                )
                infer_linear.codes.data.copy_(
                    qw.get_codes().to(infer_linear.codes.dtype)
                )
                infer_linear.scales.data.copy_(
                    qw.scales.data.to(torch.bfloat16)
                )
                if child_module.bias is not None:
                    infer_linear.bias.data.copy_(
                        child_module.bias.data.to(torch.bfloat16)
                    )

                setattr(submodule, child_name, infer_linear)

    return layer


def save_quantized_model(
    model: nn.Module, output_dir: str, source_dir: str
):
    """保存量化后的完整模型。

    保存内容包括：
    - 模型权重（safetensors 格式，自动分片）
    - 模型配置（config.json）
    - 分词器文件
    - 模型代码文件（modeling_*.py, configuration_*.py）
    - 量化配置记录（quant_config.json）

    Args:
        model: 量化后的完整模型。
        output_dir: 输出目录。
        source_dir: 源模型目录（用于复制 tokenizer/配置文件）。
    """
    logger.info(
        f"[保存] 开始保存量化模型到 {output_dir} ..."
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存模型权重（使用 safetensors）
    logger.info(
        "[保存] 保存模型权重（safetensors 格式）..."
    )
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    # 按大小分片保存（每片约 2GB 以避免单文件过大）
    max_shard_size = 2 * 1024 * 1024 * 1024  # 2GB
    shard = {}
    shard_idx = 0
    current_size = 0

    # 收集所有权重准备保存
    weight_map = {}
    all_keys = list(state_dict.keys())

    for key in tqdm(all_keys, desc="保存权重分片"):
        tensor = state_dict[key]
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and shard:
            # 保存当前分片
            shard_file = os.path.join(
                output_dir,
                f"model-{shard_idx + 1:05d}-of-00000.safetensors",
            )
            save_file(shard, shard_file)
            for k in shard:
                weight_map[k] = os.path.basename(shard_file)
            shard = {}
            shard_idx += 1
            current_size = 0
        shard[key] = tensor.contiguous().cpu()
        current_size += tensor_size

    # 保存最后的分片
    if shard:
        shard_file = os.path.join(
            output_dir,
            f"model-{shard_idx + 1:05d}-of-00000.safetensors",
        )
        save_file(shard, shard_file)
        for k in shard:
            weight_map[k] = os.path.basename(shard_file)
        shard_idx += 1

    # 更新分片文件名（重命名为带总数格式）
    total_shards = shard_idx
    for i in range(total_shards):
        old_name = os.path.join(
            output_dir,
            f"model-{i + 1:05d}-of-00000.safetensors",
        )
        new_name = os.path.join(
            output_dir,
            f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors",
        )
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            for k, v in weight_map.items():
                if v == os.path.basename(old_name):
                    weight_map[k] = os.path.basename(new_name)

    # 保存 weight_map 索引
    index_data = {"metadata": {}, "weight_map": weight_map}
    with open(
        os.path.join(output_dir, "model.safetensors.index.json"), "w"
    ) as f:
        json.dump(index_data, f, indent=2)

    logger.info(
        f"[保存] 模型权重已保存为 {total_shards} 个分片"
    )

    # 2. 复制配置文件
    logger.info("[保存] 复制模型配置和分词器文件...")
    import shutil

    files_to_copy = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "configuration_riverone_qc.py",
        "modeling_riverone_qc.py",
        "modeling_ising_vit.py",
        "conversation.py",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "video_preprocessor_config.json",
    ]

    for filename in files_to_copy:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.debug(f"  已复制: {filename}")
        else:
            logger.debug(f"  跳过（不存在）: {filename}")

    # 3. 更新 config.json 中的模型路径
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["_name_or_path"] = output_dir
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # 4. 保存量化配置信息
    quant_config = {
        "quantization_method": "AQLM",
        "scheme": "1x16",
        "in_group_size": IN_GROUP_SIZE,
        "out_group_size": OUT_GROUP_SIZE,
        "num_codebooks": NUM_CODEBOOKS,
        "nbits_per_codebook": NBITS_PER_CODEBOOK,
        "quantized_layers": f"all_{NUM_LAST_LAYERS}_of_llm",
        "quantized_components": "attention_and_mlp_linear_weights",
        "preserved_components": (
            "miniViT_vision_encoder, multimodal_projection(mlp1), "
            "embedding, lm_head, all_norm_layers"
        ),
        "source_model": "RiverOne-QC-4B-v1 (miniViT)",
    }
    with open(
        os.path.join(output_dir, "quant_config.json"), "w"
    ) as f:
        json.dump(quant_config, f, indent=2)

    logger.info(
        f"[保存] 量化模型已完整保存到: {output_dir}"
    )


# ============================================================================
# 完整性校验
# ============================================================================

def verify_quantization_integrity(
    model: nn.Module,
    target_indices: List[int],
    key_prefix: str,
):
    """校验量化范围是否正确：仅目标层的线性权重被量化，其余组件保持原始精度。

    检查项：
    1. 目标层中的线性权重是否已替换为 QuantizedLinear
    2. 非目标层是否保持原始 nn.Linear
    3. Embedding/LM Head 等是否未被量化
    4. Vision 相关组件是否未被量化（miniViT 保持原精度）

    Args:
        model: 量化后的模型。
        target_indices: 目标层索引列表。
        key_prefix: 量化器键前缀。
    """
    logger.info("=" * 60)
    logger.info("[完整性校验] 开始校验量化范围...")
    all_layers = get_layers(model)
    errors = []

    for idx, layer in enumerate(all_layers):
        is_target = idx in target_indices
        sublayers = find_sublayers(layer)

        for name, sublayer in sublayers.items():
            is_linear_like = isinstance(
                sublayer,
                (
                    nn.Linear,
                    QuantizedLinear,
                    AQLMInferenceQuantizedLinear,
                ),
            )
            is_quantizable_weight = any(
                kw in name for kw in LINEAR_LAYER_KEYWORDS
            )

            if is_target and is_quantizable_weight:
                # 目标层的线性权重必须已被量化
                if not isinstance(
                    sublayer,
                    (QuantizedLinear, AQLMInferenceQuantizedLinear),
                ):
                    errors.append(
                        f"  [错误] 目标层 {idx} 的子层 '{name}' 未被量化！"
                        f" 类型: {type(sublayer).__name__}"
                    )
            elif not is_target and is_linear_like:
                # 非目标层的线性层不应被量化
                if isinstance(
                    sublayer,
                    (QuantizedLinear, AQLMInferenceQuantizedLinear),
                ):
                    errors.append(
                        f"  [错误] 非目标层 {idx} 的子层 '{name}' "
                        f"被意外量化！"
                    )

    # 检查 Embedding / LM Head
    llm = get_llm_model(model)
    for name, module in llm.named_modules():
        if isinstance(
            module,
            (QuantizedLinear, AQLMInferenceQuantizedLinear),
        ):
            # 排除已在层内检查过的
            is_in_target_layer = any(
                f".{idx}." in name or f".layers.{idx}." in name
                for idx in target_indices
            )
            if not is_in_target_layer:
                errors.append(
                    f"  [错误] 非层内模块 '{name}' 被意外量化！"
                )

    # 检查视觉编码器（miniViT）未被量化
    if hasattr(model, "vision_model"):
        for name, module in model.vision_model.named_modules():
            if isinstance(
                module,
                (QuantizedLinear, AQLMInferenceQuantizedLinear),
            ):
                errors.append(
                    f"  [错误] vision_model 中的 '{name}' 被意外量化！"
                    f"（miniViT 应保持原精度）"
                )

    # 检查 mlp1 投影层未被量化
    if hasattr(model, "mlp1"):
        for name, module in model.mlp1.named_modules():
            if isinstance(
                module,
                (QuantizedLinear, AQLMInferenceQuantizedLinear),
            ):
                errors.append(
                    f"  [错误] mlp1 投影层中的 '{name}' 被意外量化！"
                )

    if errors:
        logger.error("[完整性校验] 发现以下问题:")
        for err in errors:
            logger.error(err)
        raise RuntimeError(
            "量化完整性校验失败！请检查量化范围配置。"
        )
    else:
        logger.info(
            "[完整性校验] 通过！量化范围正确："
            f"仅 LLM 全部 {len(target_indices)} 层线性权重被量化，"
            f"miniViT 视觉编码器、mlp1 投影层、Embedding/LM Head 等保持原精度。"
        )


# ============================================================================
# 主入口
# ============================================================================

def main():
    """RiverOne-QC-4B-v1 AQLM 量化的主执行入口（全部36层 LLM）。

    执行流程：
    1. 加载源模型（RiverOne-QC-4B-v1）
    2. 解析模型结构，定位目标层（全部36层）
    3. 准备校准数据
    4. 逐层执行 AQLM 量化
    5. 保存量化后模型
    6. 完整性校验
    """
    logger.info("=" * 60)
    logger.info(" RiverOne-QC-4B-v1 定向 AQLM 量化开始（全部36层 LLM）")
    logger.info(f" 源模型: {SOURCE_MODEL_PATH}")
    logger.info(f" 输出目录: {OUTPUT_DIR}")
    logger.info(
        f" 量化方案: AQLM 1×16 (in_group={IN_GROUP_SIZE}, "
        f"out_group={OUT_GROUP_SIZE}, "
        f"codebooks={NUM_CODEBOOKS}, nbits={NBITS_PER_CODEBOOK})"
    )
    logger.info(
        f" 量化范围: LLM 全部 {NUM_LAST_LAYERS} 层 "
        f"Attention + MLP 线性权重"
    )
    logger.info(
        f" 校准数据: {DATASET}, {NSAMPLES} 样本 × {MODEL_SEQLEN} tokens"
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: 设置默认 CUDA 设备（必须在任何 GPU 操作之前）
    # init_aq_engines 内部使用 torch.cuda.current_device() 获取设备，
    # 必须确保其返回值与 DEVICES 配置一致，否则会出现 cuda:0 vs cuda:6 的
    # 设备不匹配错误
    # ------------------------------------------------------------------
    gpu_id = int(DEVICES[0].split(":")[-1])  # 从 "cuda:6" 提取 6
    torch.cuda.set_device(gpu_id)
    logger.info(
        f"[步骤0] 已设置默认 CUDA 设备为 cuda:{gpu_id}"
    )

    # ------------------------------------------------------------------
    # Step 1: 加载源模型
    # ------------------------------------------------------------------
    logger.info("[步骤1/5] 加载源模型...")
    try:
        # 设置 sys.path 以支持 trust_remote_code
        if SOURCE_MODEL_PATH not in sys.path:
            sys.path.insert(0, SOURCE_MODEL_PATH)

        model = get_model(
            model_path=SOURCE_MODEL_PATH,
            load_quantized=None,
            dtype=DTYPE,
            device_map=None,
            attn_implementation=ATTN_IMPLEMENTATION,
            trust_remote_code=True,
        )
        model.eval()
        total_params = sum(
            p.numel() for p in model.parameters()
        )
        logger.info(
            f"[步骤1/5] 模型加载成功，"
            f"总参数量约 {total_params / 1e9:.2f}B"
        )
    except Exception as e:
        logger.error(f"[步骤1/5] 模型加载失败: {e}")
        logger.error(
            "[排查方向] 1) 确认源模型路径存在且完整 "
            "2) 确认 transformers 版本兼容 "
            "3) 确认 modeling_riverone_qc.py 在模型目录中 "
            "4) 确认依赖包已安装"
        )
        raise

    # ------------------------------------------------------------------
    # Step 2: 解析模型结构
    # ------------------------------------------------------------------
    logger.info(
        "[步骤2/5] 解析模型结构，定位量化目标..."
    )
    try:
        llm, all_layers, target_indices, index_to_name = (
            resolve_target_layers(model, NUM_LAST_LAYERS)
        )
        key_prefix = get_quantizer_key_prefix(model)
    except Exception as e:
        logger.error(
            f"[步骤2/5] 模型结构解析失败: {e}"
        )
        logger.error(
            "[排查方向] 1) 检查模型是否为支持的架构 "
            "2) 检查 LLM_KEYWORDS 是否覆盖"
        )
        raise

    # ------------------------------------------------------------------
    # Step 3: 准备校准数据
    # ------------------------------------------------------------------
    logger.info("[步骤3/5] 准备校准数据...")
    try:
        # 获取分词器
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            SOURCE_MODEL_PATH, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        devices = [torch.device(d) for d in DEVICES]
        # k-means 可使用多卡并行
        kmeans_devices = [torch.device(d) for d in KMEANS_DEVICES]
        calib_data, forward_args = prepare_calibration_data(
            model,
            tokenizer,
            NSAMPLES,
            MODEL_SEQLEN,
            DATASET,
            SEED,
            devices,
            OFFLOAD_ACTIVATIONS,
        )
    except Exception as e:
        logger.error(
            f"[步骤3/5] 校准数据准备失败: {e}"
        )
        logger.error(
            "[排查方向] 1) 检查网络连接（需下载数据集）"
            "2) 检查显存是否充足"
        )
        raise

    # ------------------------------------------------------------------
    # Step 4: 逐层执行 AQLM 量化
    # ------------------------------------------------------------------
    logger.info(
        "[步骤4/5] 开始逐层 AQLM 量化（全部36层，预计耗时较长）..."
    )

    # 构建量化参数
    args = Namespace(
        devices=kmeans_devices,  # k-means 使用所有 GPU 并行
        in_group_size=IN_GROUP_SIZE,
        out_group_size=OUT_GROUP_SIZE,
        num_codebooks=NUM_CODEBOOKS,
        nbits_per_codebook=NBITS_PER_CODEBOOK,
        attention_nbits_per_codebook=ATTENTION_NBITS_PER_CODEBOOK,
        sublayer_nbits=SUBlayer_NBITS,
        codebook_value_nbits=CODEBOOK_VALUE_NBITS,
        codebook_value_num_groups=CODEBOOK_VALUE_NUM_GROUPS,
        scale_nbits=SCALE_NBITS,
        init_max_iter=INIT_MAX_ITER,
        init_max_points_per_centroid=INIT_MAX_POINTS_PER_CENTROID,
        use_faiss=USE_FAISS,
        nredo=FAISS_NREDO,
        max_epochs=MAX_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        lr=LR,
        beam_size=BEAM_SIZE,
        relative_mse_tolerance=RELATIVE_MSE_TOLERANCE,
        print_frequency=PRINT_FREQUENCY,
        use_checkpointing=USE_CHECKPOINTING,
        true_sequential=TRUE_SEQUENTIAL,
        skip_out_loss=SKIP_OUT_LOSS,
        resume=False,
        save=None,
        on_save=None,
        finetune_max_epochs=FINETUNE_MAX_EPOCHS,
        finetune_lr=FINETUNE_LR,
        finetune_batch_size=FINETUNE_BATCH_SIZE,
        offload_activations=OFFLOAD_ACTIVATIONS,
    )

    # 初始化输入/输出缓冲区
    # inps: 当前层的输入激活值（初始为 layer 0 的输入）
    # outs: 当前层的输出激活值（将作为下一层的输入）
    inps = calib_data
    outs = [
        torch.zeros_like(inp, pin_memory=inp.is_pinned())
        for inp in inps
    ]

    # 记录总量化耗时
    total_quant_time = 0.0

    # 逐层量化（从第 0 层到第 35 层，全部36层）
    for idx in target_indices:
        layer = all_layers[idx]
        logger.info(
            f"\n{'=' * 60}\n"
            f" 量化第 {idx} 层 / 共 {len(all_layers)} 层 "
            f"(进度: {idx - target_indices[0] + 1}/{len(target_indices)})\n"
            f"{'=' * 60}"
        )

        t_layer_start = time.time()

        # 执行单层量化
        layer = quantize_single_layer(
            layer,
            idx,
            inps,
            outs,
            args,
            forward_args,
            model,
        )

        t_layer_end = time.time()
        layer_time = t_layer_end - t_layer_start
        total_quant_time += layer_time
        logger.info(
            f"[进度] 第 {idx} 层完成，"
            f"耗时 {layer_time:.1f}s，"
            f"累计 {total_quant_time:.1f}s"
        )

        # 将当前层的输出传播为下一层的输入
        if idx < target_indices[-1]:
            logger.info(
                f"[传播] 将第 {idx} 层输出传播到第 {idx + 1} 层..."
            )
            # 将输入/输出缓冲区转换为 float32 用于前向传播
            inps_float32 = [
                inp.to(
                    device=args.devices[
                        min(i, len(args.devices) - 1)
                    ],
                    dtype=torch.float32,
                )
                for i, inp in enumerate(inps)
            ]
            outs_float32 = [
                torch.zeros_like(inp) for inp in inps_float32
            ]

            # 对每层的 inps/outs 更新
            update_outs(
                layer,
                inps_float32,
                outs_float32,
                **forward_args,
            )

            # 交换缓冲区：outs 变为下一层的 inps
            inps, outs = outs_float32, [
                torch.zeros_like(out) for out in outs_float32
            ]

        # 定时清理显存
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(
        f"[步骤4/5] 全部 {len(target_indices)} 层量化完成！"
        f" 总耗时: {total_quant_time:.1f}s "
        f"({total_quant_time / 60:.1f} 分钟)"
    )

    # ------------------------------------------------------------------
    # Step 5: 转换推理格式并保存
    # ------------------------------------------------------------------
    logger.info("[步骤5/5] 转换推理格式并保存模型...")

    # 将所有训练版 QuantizedLinear 转换为推理版
    logger.info("[转换] 正在将训练版量化层转换为推理版...")
    llm_layers = get_layers(model)
    for idx in target_indices:
        llm_layers[idx] = convert_to_inference_format(
            llm_layers[idx]
        )
    logger.info("[转换] 格式转换完成")

    # 保存量化模型
    save_quantized_model(model, OUTPUT_DIR, SOURCE_MODEL_PATH)

    # 完整性校验
    verify_quantization_integrity(
        model, target_indices, key_prefix
    )

    logger.info("=" * 60)
    logger.info(" RiverOne-QC-4B-v1 AQLM 量化全部完成！")
    logger.info(f" 输出目录: {OUTPUT_DIR}")
    logger.info(f" 量化层数: {len(target_indices)} / {len(all_layers)}")
    logger.info(f" 日志文件: {LOG_FILE}")
    logger.info("=" * 60)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    # 设置环境变量以优化显存管理
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    main()
