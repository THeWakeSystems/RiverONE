#!/usr/bin/env python3
"""
=============================================================================
 RiverOne-QC-4B-v1 AQLM 量化效果验证脚本（全部36层 LLM）
=============================================================================
 脚本功能：
   - 分别加载源模型（BF16）与量化后模型（AQLM 1×16, 全部36层）
   - 使用相同 prompt 执行文本生成测试
   - 对比显存占用、推理速度、生成效果
   - 校验量化后模型结构完整性与加载正确性

 验证指标：
   - GPU 显存占用对比（加载时 / 推理时峰值）
   - 推理速度对比（tokens/秒）
   - 生成文本质量直观对比
   - 模型结构完整性检查（含 miniViT 视觉编码器保护检查）

 使用方法：
   python verify.py

 注意：
   - 需要 GPU 环境（默认 cuda:0）
   - 源模型和量化后模型路径通过脚本顶部配置区设定
   - 量化模型为全部36层 LLM 量化版本
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import gc
import json
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 添加模型路径
# ---------------------------------------------------------------------------
# 源模型路径（RiverOne-QC-4B-v1，使用 miniViT）
SOURCE_MODEL_PATH: str = "/home/hyba/lyc/RiverOne-QC-4B-v1"
# 量化后模型路径（全部36层 LLM AQLM 量化）
QUANTIZED_MODEL_PATH: str = (
    "/home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/"
    "RiverOne-QC-4B-v1-AQLM-36L"
)
if SOURCE_MODEL_PATH not in sys.path:
    sys.path.insert(0, SOURCE_MODEL_PATH)
if QUANTIZED_MODEL_PATH not in sys.path:
    sys.path.insert(0, QUANTIZED_MODEL_PATH)

# ---------------------------------------------------------------------------
# 验证配置参数
# ---------------------------------------------------------------------------
DEVICE: str = "cuda:0"              # 推理设备
DTYPE: str = "bfloat16"             # 模型加载精度
TEST_PROMPTS: List[str] = [         # 测试用 prompt 列表
    "请用中文简要介绍一下人工智能的发展历史。",
    "What is the capital of France? Please answer in one sentence.",
    "写一首关于秋天的五言绝句。",
    "Explain the difference between DNA and RNA in simple terms.",
]
MAX_NEW_TOKENS: int = 128           # 每次生成的最大新 token 数
TEMPERATURE: float = 0.7            # 生成温度
TOP_P: float = 0.9                  # nucleus sampling 参数
DO_SAMPLE: bool = False             # 是否使用采样（False=贪婪解码，结果更稳定）

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify")


# ============================================================================
# 工具函数：显存统计
# ============================================================================

def get_gpu_memory_info(
    device: str = "cuda:0",
) -> Tuple[float, float]:
    """获取当前 GPU 显存使用情况。

    Args:
        device: GPU 设备标识。

    Returns:
        (已分配显存_MB, 已缓存显存_MB)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    # MB
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    return allocated, reserved


def reset_gpu_memory():
    """清理 GPU 缓存并重置显存统计（用于精确测量）。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# 模型加载
# ============================================================================

def load_original_model() -> Optional[nn.Module]:
    """加载原始 RiverOne-QC-4B-v1 源模型。

    Returns:
        加载成功的模型实例，失败返回 None。
    """
    logger.info("=" * 60)
    logger.info("[加载] 正在加载原始模型（RiverOne-QC-4B-v1）...")
    try:
        from transformers import AutoModel

        # 先记录加载前显存
        reset_gpu_memory()
        mem_before, _ = get_gpu_memory_info(DEVICE)

        # 使用 AutoModel（而非直接 import），避免 modeling 文件中的
        # 相对导入 (from .xxx import) 报 "no known parent package" 错误
        model = AutoModel.from_pretrained(
            SOURCE_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=getattr(torch, DTYPE),
            device_map=None,
        )
        model = model.to(DEVICE)
        model.eval()

        mem_after, _ = get_gpu_memory_info(DEVICE)
        mem_used = mem_after - mem_before

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        logger.info(f"[加载] 原始模型加载成功！")
        logger.info(f"  总参数量:      {total_params / 1e9:.2f}B")
        logger.info(
            f"  可训练参数:    {trainable_params / 1e9:.2f}B"
        )
        logger.info(f"  加载显存占用:  {mem_used:.1f} MB")

        return model
    except Exception as e:
        logger.error(f"[加载] 原始模型加载失败: {e}")
        logger.error(
            "[排查] 1) 确认模型路径存在 2) 确认依赖已安装 "
            "3) 确认 GPU 显存充足"
        )
        return None


def load_quantized_model() -> Optional[nn.Module]:
    """加载 AQLM 量化后的 RiverOne-QC-4B-v1 模型（全部36层）。

    自动识别 .codes/.codebooks/.scales 量化权重，
    将 nn.Linear 替换为 aqlm.inference.QuantizedLinear。

    Returns:
        加载成功的量化模型实例，失败返回 None。
    """
    logger.info("=" * 60)
    logger.info(
        "[加载] 正在加载量化后模型（AQLM 1×16, 全部36层 LLM）..."
    )
    try:
        from collections import defaultdict
        from safetensors.torch import load_file
        from aqlm import QuantizedLinear as AQLMLinear
        from transformers import AutoModel

        reset_gpu_memory()
        mem_before, _ = get_gpu_memory_info(DEVICE)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import logging as hf_logging

            hf_logging.set_verbosity_error()
            # 使用 AutoModel 而非直接 import，避免相对导入报错
            model = AutoModel.from_pretrained(
                QUANTIZED_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=getattr(torch, DTYPE),
                device_map=None,
            )
            hf_logging.set_verbosity_warning()
        model = model.to(DEVICE)

        # ── 加载 AQLM 量化权重 ────────────────────────────
        quant_config_path = os.path.join(
            QUANTIZED_MODEL_PATH, "quant_config.json"
        )
        if os.path.exists(quant_config_path):
            with open(quant_config_path) as f:
                quant_config = json.load(f)
            if (
                quant_config.get("quantization_method") == "AQLM"
            ):
                index_path = os.path.join(
                    QUANTIZED_MODEL_PATH,
                    "model.safetensors.index.json",
                )
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]

                quantized_groups = defaultdict(dict)
                for key in weight_map:
                    if key.endswith(".codebooks"):
                        quantized_groups[key[:-10]][
                            "codebooks_key"
                        ] = key
                    elif key.endswith(".codes"):
                        quantized_groups[key[:-6]][
                            "codes_key"
                        ] = key
                    elif key.endswith(".scales"):
                        quantized_groups[key[:-7]][
                            "scales_key"
                        ] = key

                # 加载所有分片
                all_tensors = {}
                for shard in sorted(
                    set(weight_map.values())
                ):
                    path = os.path.join(
                        QUANTIZED_MODEL_PATH, shard
                    )
                    if os.path.exists(path):
                        all_tensors.update(load_file(path))

                llm_layers = (
                    model.language_model.model.layers
                )
                for base, info in quantized_groups.items():
                    parts = base.split(".")
                    layer_idx = int(parts[3])
                    sub_path = parts[4:]
                    cb = all_tensors[
                        info["codebooks_key"]
                    ]
                    cd = all_tensors[info["codes_key"]]
                    sc = all_tensors[info["scales_key"]]
                    bias = all_tensors.get(
                        f"{base}.bias", None
                    )
                    nc, cs, og, ig = cb.shape
                    nbits = cs.bit_length() - 1

                    ql = AQLMLinear(
                        cd.shape[1] * ig,
                        cd.shape[0] * og,
                        ig,
                        og,
                        nc,
                        nbits,
                        bias=bias is not None,
                        dtype=cb.dtype,
                    )
                    ql.codebooks.data.copy_(cb)
                    ql.codes.data.copy_(
                        cd.to(ql.codes.dtype)
                    )
                    ql.scales.data.copy_(sc)
                    if bias is not None:
                        ql.bias.data.copy_(bias)

                    parent = llm_layers[layer_idx]
                    for seg in sub_path[:-1]:
                        parent = getattr(parent, seg)
                    ql = ql.to(
                        device=next(
                            parent.parameters()
                        ).device
                    )
                    setattr(
                        parent, sub_path[-1], ql
                    )

                logger.info(
                    f"  [AQLM] 已加载 "
                    f"{len(quantized_groups)} 个量化层"
                )

        model.eval()

        mem_after, _ = get_gpu_memory_info(DEVICE)
        mem_used = mem_after - mem_before

        total_params = sum(
            p.numel() for p in model.parameters()
        )

        logger.info(f"[加载] 量化模型加载成功！")
        logger.info(
            f"  总参数量:      {total_params / 1e9:.2f}B"
        )
        logger.info(
            f"  加载显存占用:  {mem_used:.1f} MB"
        )

        return model
    except Exception as e:
        logger.error(
            f"[加载] 量化模型加载失败: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# 结构完整性检查
# ============================================================================

def check_model_structure(
    model: nn.Module, model_name: str
) -> bool:
    """检查模型结构是否完整：验证关键组件存在且类型正确。

    检查项：
    - vision_model（miniViT）存在且未被量化
    - language_model 存在
    - mlp1 投影层存在且未被量化
    - LLM 层数量正确（36层）
    - 量化层统计

    Args:
        model: 模型实例。
        model_name: 模型名称（用于日志）。

    Returns:
        结构完整性检查是否通过。
    """
    logger.info(
        f"[结构检查] 检查 {model_name} 模型结构..."
    )
    all_ok = True

    # 1. 检查核心组件
    expected_attrs = {
        "vision_model": "视觉编码器（miniViT）",
        "language_model": "语言模型（Qwen3-4B）",
        "mlp1": "多模态投影层",
    }
    for attr, desc in expected_attrs.items():
        if not hasattr(model, attr):
            logger.error(
                f"  [错误] 缺少 {desc} ({attr})"
            )
            all_ok = False
        else:
            logger.info(
                f"  [通过] {desc} ({attr}) 存在"
            )

    # 2. 检查 vision_model（miniViT）未被量化
    if hasattr(model, "vision_model"):
        from aqlm import QuantizedLinear as AQLMLinear

        for name, module in model.vision_model.named_modules():
            if isinstance(module, AQLMLinear):
                logger.error(
                    f"  [错误] vision_model 中的 {name} "
                    f"被量化！（miniViT 应保留原精度）"
                )
                all_ok = False
        if all_ok:
            logger.info(
                f"  [通过] vision_model（miniViT）未被量化"
            )

    # 3. 检查 mlp1 投影层未被量化
    if hasattr(model, "mlp1"):
        from aqlm import QuantizedLinear as AQLMLinear

        for name, module in model.mlp1.named_modules():
            if isinstance(module, AQLMLinear):
                logger.error(
                    f"  [错误] mlp1 投影层中的 {name} "
                    f"被量化！（应保留原精度）"
                )
                all_ok = False
        if all_ok:
            logger.info(
                f"  [通过] mlp1 投影层未被量化"
            )

    # 4. 统计 LLM 层数和量化层数
    if hasattr(model, "language_model"):
        llm = model.language_model
        if hasattr(llm, "model") and hasattr(
            llm.model, "layers"
        ):
            total_layers = len(llm.model.layers)
            quantized_count = 0
            from aqlm import (
                QuantizedLinear as AQLMLinear,
            )

            for i, layer in enumerate(
                llm.model.layers
            ):
                has_quantized = False
                for module in layer.modules():
                    if isinstance(module, AQLMLinear):
                        has_quantized = True
                        break
                if has_quantized:
                    quantized_count += 1
                    logger.info(
                        f"  [量化层] layer {i}: 已量化"
                    )
            logger.info(
                f"  [统计] LLM 总层数: {total_layers}, "
                f"量化层数: {quantized_count}"
            )
            # 全部36层都应被量化
            if quantized_count == 36:
                logger.info(
                    f"  [通过] 量化层数正确（全部 36 层）"
                )
            else:
                logger.warning(
                    f"  [警告] 量化层数为 {quantized_count}，"
                    f"预期 36 层"
                )

    # 5. 检查 Embedding / LM Head 未被量化
    if hasattr(model, "language_model"):
        llm = model.language_model
        from aqlm import QuantizedLinear as AQLMLinear

        # 检查 embed_tokens
        if hasattr(llm, "model") and hasattr(
            llm.model, "embed_tokens"
        ):
            if isinstance(
                llm.model.embed_tokens, AQLMLinear
            ):
                logger.error(
                    f"  [错误] Embedding 层被量化！"
                )
                all_ok = False
            else:
                logger.info(
                    f"  [通过] Embedding 层未被量化"
                )
        # 检查 lm_head
        if hasattr(llm, "lm_head"):
            if isinstance(llm.lm_head, AQLMLinear):
                logger.error(
                    f"  [错误] LM Head 被量化！"
                )
                all_ok = False
            else:
                logger.info(
                    f"  [通过] LM Head 未被量化"
                )

    if all_ok:
        logger.info(
            f"[结构检查] {model_name} 全部通过！"
        )
    else:
        logger.error(
            f"[结构检查] {model_name} 存在问题！"
        )
    return all_ok


# ============================================================================
# 推理测试
# ============================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    do_sample: bool = DO_SAMPLE,
    device: str = DEVICE,
) -> Tuple[str, float, float]:
    """对单个 prompt 执行推理，返回生成文本和性能指标。

    Args:
        model: 模型实例。
        tokenizer: 分词器。
        prompt: 输入提示文本。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度。
        top_p: nucleus sampling 参数。
        do_sample: 是否采样。
        device: 推理设备。

    Returns:
        (generated_text, inference_time_seconds, peak_gpu_memory_mb)
    """
    # Tokenize
    inputs = tokenizer(
        prompt, return_tensors="pt"
    ).to(device)

    # 预热：先跑一次小前向（不计时）
    try:
        _ = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
            or tokenizer.eos_token_id,
        )
    except Exception:
        pass  # 预热失败不影响正式测试

    torch.cuda.synchronize()
    reset_gpu_memory()

    # 正式推理计时
    t_start = time.time()
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id
            or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    except Exception as e:
        # 兼容某些模型不支持 sampling 参数
        logger.warning(
            f"生成时遇到问题，回退到基础配置: {e}"
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
            or tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    t_end = time.time()
    inference_time = t_end - t_start

    # 显存统计
    peak_mem = (
        torch.cuda.max_memory_allocated(device)
        / (1024**2)
    )

    # 解码（仅新生成的 token）
    generated_ids = outputs[0][
        inputs.input_ids.shape[1] :
    ]
    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )

    return generated_text, inference_time, peak_mem


def run_chat_inference(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, float, float]:
    """使用 model.chat() 执行纯文本推理，返回文本和性能指标。

    统一使用 model.chat()（内部走 chat template），
    原版和量化模型均适用。

    Args:
        model: RiverOneQCModel 实例。
        tokenizer: 分词器。
        prompt: 输入提示。
        max_new_tokens: 最大生成 token 数。

    Returns:
        (generated_text, time_seconds, peak_mem_mb)
    """
    torch.cuda.synchronize()
    reset_gpu_memory()

    t_start = time.time()
    try:
        response, _ = model.chat(
            tokenizer=tokenizer,
            pixel_values=None,  # 纯文本推理
            question=prompt,
            generation_config={
                "max_new_tokens": max_new_tokens,
                "do_sample": DO_SAMPLE,
                "temperature": (
                    TEMPERATURE if DO_SAMPLE else 1.0
                ),
                "top_p": TOP_P if DO_SAMPLE else 1.0,
            },
            return_history=True,
        )
    except Exception as e:
        logger.warning(
            f"model.chat 失败: {e}，"
            f"回退 language_model.generate"
        )
        inputs = tokenizer(
            prompt, return_tensors="pt"
        ).to(DEVICE)
        outputs = model.language_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][
                inputs["input_ids"].shape[1] :
            ],
            skip_special_tokens=True,
        )

    torch.cuda.synchronize()
    t_end = time.time()
    inference_time = t_end - t_start
    peak_mem = (
        torch.cuda.max_memory_allocated(DEVICE)
        / (1024**2)
    )

    return response, inference_time, peak_mem


# ============================================================================
# 主入口
# ============================================================================

def main():
    """验证主入口：加载模型 → 结构检查 → 推理测试 → 对比报告。"""
    logger.info("=" * 60)
    logger.info(
        " RiverOne-QC-4B-v1 AQLM 量化效果验证"
    )
    logger.info(
        f" 源模型:     {SOURCE_MODEL_PATH}"
    )
    logger.info(
        f" 量化模型:   {QUANTIZED_MODEL_PATH}"
    )
    logger.info(f" 设备:       {DEVICE}")
    logger.info(f" 精度:       {DTYPE}")
    logger.info("=" * 60)

    # ── 1. 检查量化模型目录是否存在 ──────────────────────
    if not os.path.exists(QUANTIZED_MODEL_PATH):
        logger.error(
            f"[错误] 量化模型目录不存在: "
            f"{QUANTIZED_MODEL_PATH}"
        )
        logger.error(
            "请先运行 quantize.py 完成量化后再验证。"
        )
        sys.exit(1)

    # ── 2. 加载分词器 ────────────────────────────────────
    from transformers import AutoTokenizer

    logger.info("[加载] 正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. 加载原始模型 ──────────────────────────────────
    original_model = load_original_model()
    if original_model is None:
        logger.error("原始模型加载失败，无法进行对比测试。")
        sys.exit(1)

    # ── 4. 原始模型结构检查 ──────────────────────────────
    check_model_structure(original_model, "原始模型")

    # ── 5. 原始模型推理测试 ──────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[推理测试] 原始模型推理测试...")
    original_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        logger.info(
            f"\n--- 原始模型 Prompt {i + 1} ---"
        )
        logger.info(f"输入: {prompt[:80]}...")

        text, inf_time, peak_mem = run_chat_inference(
            original_model, tokenizer, prompt
        )
        new_tokens = len(
            tokenizer.encode(text)
        )
        tokens_per_sec = (
            new_tokens / inf_time if inf_time > 0 else 0
        )

        logger.info(f"输出: {text[:120]}...")
        logger.info(
            f"耗时: {inf_time:.2f}s | "
            f"新token数: {new_tokens} | "
            f"速度: {tokens_per_sec:.1f} tok/s | "
            f"峰值显存: {peak_mem:.0f} MB"
        )

        original_results.append(
            {
                "prompt": prompt,
                "text": text,
                "time_s": inf_time,
                "new_tokens": new_tokens,
                "tokens_per_sec": tokens_per_sec,
                "peak_mem_mb": peak_mem,
            }
        )

    # 释放原始模型显存
    del original_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 6. 加载量化模型 ──────────────────────────────────
    quantized_model = load_quantized_model()
    if quantized_model is None:
        logger.error("量化模型加载失败，无法进行对比测试。")
        sys.exit(1)

    # ── 7. 量化模型结构检查 ──────────────────────────────
    check_model_structure(quantized_model, "量化模型")

    # ── 8. 量化模型推理测试 ──────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[推理测试] 量化模型推理测试...")
    quantized_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        logger.info(
            f"\n--- 量化模型 Prompt {i + 1} ---"
        )
        logger.info(f"输入: {prompt[:80]}...")

        text, inf_time, peak_mem = run_chat_inference(
            quantized_model, tokenizer, prompt
        )
        new_tokens = len(
            tokenizer.encode(text)
        )
        tokens_per_sec = (
            new_tokens / inf_time if inf_time > 0 else 0
        )

        logger.info(f"输出: {text[:120]}...")
        logger.info(
            f"耗时: {inf_time:.2f}s | "
            f"新token数: {new_tokens} | "
            f"速度: {tokens_per_sec:.1f} tok/s | "
            f"峰值显存: {peak_mem:.0f} MB"
        )

        quantized_results.append(
            {
                "prompt": prompt,
                "text": text,
                "time_s": inf_time,
                "new_tokens": new_tokens,
                "tokens_per_sec": tokens_per_sec,
                "peak_mem_mb": peak_mem,
            }
        )

    # ── 9. 对比报告 ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(" 对比报告：原始模型 vs 量化模型（全部36层）")
    logger.info("=" * 60)

    # 汇总统计
    orig_avg_speed = (
        sum(r["tokens_per_sec"] for r in original_results)
        / len(original_results)
        if original_results
        else 0
    )
    quant_avg_speed = (
        sum(
            r["tokens_per_sec"]
            for r in quantized_results
        )
        / len(quantized_results)
        if quantized_results
        else 0
    )
    orig_avg_mem = (
        sum(r["peak_mem_mb"] for r in original_results)
        / len(original_results)
        if original_results
        else 0
    )
    quant_avg_mem = (
        sum(
            r["peak_mem_mb"]
            for r in quantized_results
        )
        / len(quantized_results)
        if quantized_results
        else 0
    )

    logger.info(
        f"\n{'指标':<20} {'原始模型':<20} {'量化模型':<20} {'变化':<15}"
    )
    logger.info("-" * 75)
    logger.info(
        f"{'平均推理速度':<20} "
        f"{orig_avg_speed:>8.1f} tok/s    "
        f"{quant_avg_speed:>8.1f} tok/s    "
        f"{'↓' + str(round((1 - quant_avg_speed / orig_avg_speed) * 100, 1)) + '%' if orig_avg_speed > 0 else 'N/A':>15}"
    )
    logger.info(
        f"{'平均峰值显存':<20} "
        f"{orig_avg_mem:>8.0f} MB      "
        f"{quant_avg_mem:>8.0f} MB      "
        f"{'↓' + str(round((1 - quant_avg_mem / orig_avg_mem) * 100, 1)) + '%' if orig_avg_mem > 0 else 'N/A':>15}"
    )

    # 生成质量对比（逐 prompt）
    logger.info(f"\n{'─' * 75}")
    logger.info(" 生成质量对比:")
    for i in range(len(TEST_PROMPTS)):
        logger.info(f"\n Prompt {i + 1}: {TEST_PROMPTS[i][:60]}...")
        logger.info(
            f"   原始: {original_results[i]['text'][:100]}..."
        )
        logger.info(
            f"   量化: {quantized_results[i]['text'][:100]}..."
        )

    # 保存验证结果
    results = []
    for i in range(len(TEST_PROMPTS)):
        results.append(
            {
                "prompt": TEST_PROMPTS[i],
                "original": original_results[i],
                "quantized": quantized_results[i],
            }
        )

    summary = {
        "source_model": SOURCE_MODEL_PATH,
        "quantized_model": QUANTIZED_MODEL_PATH,
        "avg_original_speed_tok_s": orig_avg_speed,
        "avg_quantized_speed_tok_s": quant_avg_speed,
        "avg_original_mem_mb": orig_avg_mem,
        "avg_quantized_mem_mb": quant_avg_mem,
        "results": results,
    }

    verify_results_path = os.path.join(
        QUANTIZED_MODEL_PATH, "verify_results.json"
    )
    with open(verify_results_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(
        f"\n验证结果已保存到: {verify_results_path}"
    )

    logger.info(
        "\n" + "=" * 60
    )
    logger.info(" RiverOne-QC-4B-v1 AQLM 验证完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
