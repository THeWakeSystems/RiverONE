#!/usr/bin/env python3
"""
=============================================================================
RiverOne 正式发布 AQLM 量化入口 — 2×16 scheme, L8-L32 (25层), MLP + Attention
=============================================================================
 ★ 量化范围: L8-L32 (第8至第32层, 共25层), MLP + Attention ★
 ★ 保留 bf16: L0-L7, L33-L35, ViT, embedding, lm_head, norms ★
 ★ 校准数据: QcalEval ZS-SFT (多模态: 图像→IsingViT→LLM hidden states) ★

方案定义:
  num_codebooks=2, nbits_per_codebook=16, in_group_size=16, out_group_size=1
  codebook_size=65536, 等效位宽 ~1 bit/param
  GPU: cuda:0
  源模型: riverone-4b-mpo (RiverOne-QC-4B-MPO)

使用方法:
  python quantize.py

输出:
  ../weights/RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP/
=============================================================================
"""
from __future__ import annotations

import os
import sys
import json

import torch
import torch.nn as nn
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# 将框架加入路径
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import _framework as qm

# ═══════════════════════════════════════════════════════════════
# 量化范围: L8-L32 (总共25层, MLP + Attention)
# ═══════════════════════════════════════════════════════════════

TARGET_FIRST_LAYER = 8   # L8 (0-indexed)
TARGET_LAST_LAYER = 32   # L32 (0-indexed, inclusive)

# ═══════════════════════════════════════════════════════════════
# 配置 (覆盖 _framework 默认值)
# ═══════════════════════════════════════════════════════════════

qm.SOURCE_MODEL_PATH = os.environ.get(
    "RIVERONE_SOURCE_MODEL",
    "/home/lxy/workspace/riverone-release/RiverOne-QC-4B-v2",
)
qm.OUTPUT_DIR = os.environ.get(
    "RIVERONE_OUTPUT_DIR",
    os.path.join(PROJECT_DIR, "weights", "RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP"),
)
qm.LOG_FILE = os.path.join(SCRIPT_DIR, "quantize.log")

qm.NUM_CODEBOOKS = 2
qm.NBITS_PER_CODEBOOK = 16
qm.NSAMPLES = 64
qm.MODEL_SEQLEN = 2048
qm.OFFLOAD_ACTIVATIONS = False
qm.USE_FAISS = True
qm.INIT_MAX_ITER = 100
qm.INIT_MAX_POINTS_PER_CENTROID = 5
qm.LINEAR_LAYER_KEYWORDS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

CALIBRATION_JSONL = os.environ.get(
    "RIVERONE_CALIBRATION_JSONL",
    "/home/lxy/workspace/datasets/vqa_format/qcaleval_zs_sft.jsonl",
)
IMAGE_BASE = os.environ.get(
    "RIVERONE_IMAGE_BASE",
    "/home/lxy/workspace/datasets/vqa_format",
)

qm.logger = qm.setup_logging(qm.LOG_FILE)


# ═══════════════════════════════════════════════════════════════
# 覆写 resolve_target_layers: 限定 L8-L32
# ═══════════════════════════════════════════════════════════════

def resolve_target_layers(model, num_last_layers):
    """覆写: 返回 L8-L32 而非最后N层"""
    qm.logger.info("=" * 60)
    qm.logger.info("[模型解析] 开始解析模型结构...")

    llm = qm.locate_language_model(model)
    qm.logger.info(f"[模型解析] LLM 分支类型: {type(llm).__name__}")
    llm_config = qm.get_llm_config(model)
    qm.logger.info(
        f"[模型解析] LLM 配置: hidden_size={llm_config.hidden_size}, "
        f"num_hidden_layers={llm_config.num_hidden_layers}"
    )

    all_layers = qm.get_layers(model)
    total_layers = len(all_layers)
    qm.logger.info(f"[模型解析] Transformer 总层数: {total_layers}")

    target_indices = list(range(TARGET_FIRST_LAYER, TARGET_LAST_LAYER + 1))
    qm.logger.info(
        f"[模型解析] 目标量化层索引: {target_indices} "
        f"(L{TARGET_FIRST_LAYER}-L{TARGET_LAST_LAYER}, 共 {len(target_indices)} 层)"
    )

    key_prefix = qm.get_quantizer_key_prefix(model)
    qm.logger.info(f"[模型解析] 量化器键前缀: '{key_prefix}'")

    qm.logger.info("-" * 60)
    qm.logger.info("[层级匹配清单] 以下为待量化的 L8-L32 层及其子模块:")
    index_to_name = {}
    from _framework import find_sublayers as _find_sublayers
    for idx in target_indices:
        layer = all_layers[idx]
        sublayer_names = list(_find_sublayers(layer).keys())
        quantizable = [
            n for n in sublayer_names
            if any(kw in n for kw in qm.LINEAR_LAYER_KEYWORDS)
        ]
        layer_path = f"{key_prefix}.{idx}"
        index_to_name[idx] = layer_path
        qm.logger.info(
            f"  层 {idx:2d} ({layer_path}): 共 {len(sublayer_names)} 个子层, "
            f"可量化线性层 {len(quantizable)} 个 -> {quantizable}"
        )

    qm.logger.info("-" * 60)
    qm.logger.info("[排除清单] 以下组件保持原始精度，不会被量化:")
    excluded_components = []
    for name, _ in model.named_parameters():
        is_target = any(
            f"{key_prefix}.{idx}" in name for idx in target_indices
        )
        is_quantizable_weight = any(kw in name for kw in qm.LINEAR_LAYER_KEYWORDS)
        if is_target and is_quantizable_weight:
            continue
        if "weight" in name:
            excluded_components.append(name)
    for comp in excluded_components[:20]:
        qm.logger.info(f"  [保留] {comp}")
    if len(excluded_components) > 20:
        qm.logger.info(f"  ... 及另外 {len(excluded_components) - 20} 个参数")

    qm.logger.info("=" * 60)
    return llm, all_layers, target_indices, index_to_name


# ═══════════════════════════════════════════════════════════════
# GPU-safe update_outs (GPU0)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def update_outs(layer, inps, outs, **forward_args):
    gpu_device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device=gpu_device)
    layer_dtype = next(layer.parameters()).dtype

    rotary_emb = forward_args.pop("rotary_emb", None)
    default_pos_ids = forward_args.pop("default_position_ids", None)

    for i, inp_tensor in enumerate(inps):
        inp_tensor = inp_tensor.to(device=gpu_device)
        seq_len = inp_tensor.shape[1]
        if default_pos_ids is not None and rotary_emb is not None:
            pos_ids = default_pos_ids[:, :seq_len].to(gpu_device)
        else:
            pos_ids = torch.arange(seq_len, device=gpu_device).unsqueeze(0)

        for j in range(len(inp_tensor)):
            x = inp_tensor[j].to(device=gpu_device, dtype=layer_dtype).unsqueeze(0)
            layer_kwargs = {}
            if rotary_emb is not None:
                cos, sin = rotary_emb(x, pos_ids)
                layer_kwargs["position_embeddings"] = (cos, sin)
            out = layer(x, **layer_kwargs)[0]
            outs[i][j].copy_(out.reshape_as(outs[i][j]))

    if rotary_emb is not None:
        forward_args["rotary_emb"] = rotary_emb
    if default_pos_ids is not None:
        forward_args["default_position_ids"] = default_pos_ids


# ═══════════════════════════════════════════════════════════════
# 量化后保持 GPU (无坍塌层跳过 — L8-L32 全域量化)
# ═══════════════════════════════════════════════════════════════

_orig_qsl = qm.quantize_single_layer

def quantize_single_layer(layer, layer_idx, inps, outs, args, forward_args, model):
    """量化完成后将层保留在 GPU 上"""
    result = _orig_qsl(layer, layer_idx, inps, outs, args, forward_args, model)
    return result.to(device=args.devices[0])


# ═══════════════════════════════════════════════════════════════
# 完整性校验
# ═══════════════════════════════════════════════════════════════

_orig_verify = qm.verify_quantization_integrity

def verify_quantization_integrity(model, target_indices, key_prefix):
    """校验 L8-L32 MLP+Attn 层已量化，其余保持原始精度"""
    from aqlm import QuantizedLinear as AQLMInferenceQuantizedLinear
    QuantizedLinear = AQLMInferenceQuantizedLinear

    errors = []
    llm = qm.get_llm_model(model)

    for idx in target_indices:
        layer_path = f"{key_prefix}.{idx}"
        # Navigate to layer using model.language_model.model.layers[idx]
        target_layer = llm
        for part in layer_path.split(".")[1:]:
            if hasattr(target_layer, part):
                target_layer = getattr(target_layer, part)
            elif part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = None
                break

        if target_layer is None:
            continue

        for name, sublayer in target_layer.named_modules():
            sublayer_type = type(sublayer)
            is_linear_like = sublayer_type in (
                nn.Linear, QuantizedLinear, AQLMInferenceQuantizedLinear,
            )
            is_quantizable = any(kw in name for kw in qm.LINEAR_LAYER_KEYWORDS)

            if is_quantizable:
                is_quantized = sublayer_type in (QuantizedLinear, AQLMInferenceQuantizedLinear)
                if not is_quantized:
                    errors.append(
                        f"  [错误] 目标层 {idx} 的子层 '{name}' 未被量化！"
                        f" 类型: {sublayer_type.__name__}"
                    )
            elif not is_quantizable and is_linear_like:
                if sublayer_type in (QuantizedLinear, AQLMInferenceQuantizedLinear):
                    errors.append(
                        f"  [错误] 非目标层 {idx} 的子层 '{name}' 被意外量化！"
                    )

    # Embedding / LM Head check
    for name, module in llm.named_modules():
        if isinstance(module, (QuantizedLinear, AQLMInferenceQuantizedLinear)):
            is_in_target = any(
                f".{i}." in name or f".layers.{i}." in name
                for i in target_indices
            )
            if not is_in_target:
                errors.append(f"  [错误] 非层内模块 '{name}' 被意外量化！")

    # Vision encoder check
    if hasattr(model, "vision_model"):
        for name, module in model.vision_model.named_modules():
            if isinstance(module, (QuantizedLinear, AQLMInferenceQuantizedLinear)):
                errors.append(f"  [错误] vision_model 中的 '{name}' 被意外量化！")

    # mlp1 check
    if hasattr(model, "mlp1"):
        for name, module in model.mlp1.named_modules():
            if isinstance(module, (QuantizedLinear, AQLMInferenceQuantizedLinear)):
                errors.append(f"  [错误] mlp1 投影层中的 '{name}' 被意外量化！")

    if errors:
        qm.logger.error("[完整性校验] 发现以下问题:")
        for err in errors:
            qm.logger.error(err)
        raise RuntimeError("量化完整性校验失败！")
    else:
        qm.logger.info(
            f"[完整性校验] 通过！L8-L32 ({len(target_indices)} 层) MLP+Attn 量化完成，"
            f"miniViT/mlp1/Embedding/LM Head/Norms 保持原精度。"
        )


# ═══════════════════════════════════════════════════════════════
# ★ 多模态校准数据收集 ★
# ═══════════════════════════════════════════════════════════════

class _CatcherExit(Exception):
    pass

class _Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.captured = None

    def forward(self, inp, **kwargs):
        self.captured = inp.detach()
        raise _CatcherExit()


def _multimodal_collect_layer_inputs(model, tokenizer, nsamples, seqlen, devices, offload_activations):
    from transformers import AutoProcessor

    qm.logger.info(f"[校准数据] 多模态模式: 图像→IsingViT→LLM hidden states")
    qm.logger.info(f"[校准数据] JSONL: {CALIBRATION_JSONL}")
    qm.logger.info(f"[校准数据] 图像目录: {IMAGE_BASE}")

    device = devices[0]

    proc = AutoProcessor.from_pretrained(qm.SOURCE_MODEL_PATH, trust_remote_code=True)

    entries = []
    with open(CALIBRATION_JSONL, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    qm.logger.info(f"[校准数据] 共 {len(entries)} 条 JSONL 条目")

    layers = qm.get_layers(model)
    hidden_size = qm.get_hidden_size(model)
    layer_device_original = next(layers[0].parameters()).device
    model_dtype = next(model.parameters()).dtype

    layers[0] = layers[0].to(device)
    catcher = _Catcher(layers[0])
    layers[0] = catcher

    model = model.to(device)

    qm.logger.info(f"[校准数据] 开始逐条处理多模态样本...")
    qm.logger.info(f"[校准数据] 目标: {nsamples} 个样本 × {seqlen} tokens")

    inps_list = []
    img_load_errors = 0
    skipped = 0

    from tqdm import tqdm

    for entry in tqdm(entries, desc="多模态校准"):
        if len(inps_list) >= nsamples:
            break

        text_parts = []
        for turn in entry.get("conversations", []):
            content = turn.get("content", "")
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            if content and isinstance(content, str):
                text_parts.append(content)

        if not text_parts:
            skipped += 1
            continue

        text = "\n".join(text_parts)

        img_field = entry.get("image", "")
        if isinstance(img_field, (list, tuple)):
            images_raw = img_field
        else:
            images_raw = [img_field]

        image_token_count = text.count("<image>")
        if image_token_count == 0:
            skipped += 1
            continue

        text = text.replace("<image>", proc.image_token)

        if len(images_raw) > image_token_count:
            images_raw = images_raw[:image_token_count]

        images = []
        for rel_path in images_raw:
            if isinstance(rel_path, bytes):
                rel_path = rel_path.decode("utf-8")
            img_path = os.path.join(IMAGE_BASE, rel_path)
            if not os.path.exists(img_path):
                break
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                break

        if len(images) != image_token_count:
            img_load_errors += 1
            continue

        try:
            if len(images) == 1:
                images = images[0]
            inputs = proc(
                text=text, images=images, return_tensors="pt",
                max_length=seqlen * 4, truncation=True,
            )
        except Exception:
            skipped += 1
            continue

        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=model_dtype)

        try:
            with torch.no_grad():
                model(input_ids=input_ids, pixel_values=pixel_values)
        except _CatcherExit:
            hidden = catcher.captured
            actual_len = hidden.shape[1]
            if actual_len >= seqlen:
                hidden = hidden[:, :seqlen, :]
            else:
                pad = torch.zeros(
                    1, seqlen - actual_len, hidden_size,
                    dtype=hidden.dtype, device=device
                )
                hidden = torch.cat([hidden, pad], dim=1)
            inps_list.append(hidden[0])
        except Exception:
            skipped += 1
            continue

    layers[0] = catcher.module
    layers[0] = layers[0].to(layer_device_original)

    qm.logger.info(
        f"[校准数据] 收集完成: {len(inps_list)}/{nsamples} 样本, "
        f"图像加载失败: {img_load_errors}, 跳过: {skipped}"
    )

    if len(inps_list) < nsamples:
        qm.logger.warning(
            f"[校准数据] 样本不足 ({len(inps_list)} < {nsamples}), 将重复使用已有样本"
        )
        while len(inps_list) < nsamples:
            inps_list.append(inps_list[len(inps_list) % max(1, len(inps_list))])

    inps_list = inps_list[:nsamples]

    nsamples_per_device = (nsamples - 1) // len(devices) + 1
    inps = []
    for d in range(len(devices)):
        start = d * nsamples_per_device
        end = min(start + nsamples_per_device, nsamples)
        batch = torch.stack(inps_list[start:end])
        batch = batch.to(device=devices[d] if not offload_activations else "cpu")
        inps.append(batch)

    rotary_emb = qm.get_rotary_emb_module(model)
    default_pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    forward_args = {
        "rotary_emb": rotary_emb,
        "default_position_ids": default_pos_ids,
    }

    qm.logger.info(f"[校准数据] 最终 inps: {[t.shape for t in inps]}")
    return inps, forward_args


def prepare_calibration_data(model, tokenizer, nsamples, seqlen, dataset_name, seed, devices, offload_activations):
    return _multimodal_collect_layer_inputs(model, tokenizer, nsamples, seqlen, devices, offload_activations)


# ═══════════════════════════════════════════════════════════════
# 修正 quant_config.json
# ═══════════════════════════════════════════════════════════════

_original_save = qm.save_quantized_model

def save_quantized_model(model, output_dir, source_dir):
    _original_save(model, output_dir, source_dir)
    qc_path = os.path.join(output_dir, "quant_config.json")
    if os.path.exists(qc_path):
        with open(qc_path, "r") as f:
            qc = json.load(f)
        qc["scheme"] = "2x16-AttnMLP-L8L32"
        qc["num_codebooks"] = 2
        qc["nbits_per_codebook"] = 16
        qc["in_group_size"] = 16
        qc["out_group_size"] = 1
        qc["quantized_layers"] = "L8-L32 (25 of 36, middle layers, MLP + Attention)"
        qc["quantized_components"] = "attn_qkv_o + mlp_gate_up_down (L8-L32)"
        qc["preserved_components"] = "layers_0..7, layers_33..35, miniViT, embedding, lm_head, norms"
        qc["source_model"] = "RiverOne-QC-4B-MPO (IsingViT)"
        qc["kmeans"] = "FAISS k-means++, 100 iter, max_points_per_centroid=5"
        qc["calibration_data"] = "qcaleval_zs_sft.jsonl (multimodal: image→IsingViT→LLM)"
        qc["init_max_points_per_centroid"] = 5
        num_layers = TARGET_LAST_LAYER - TARGET_FIRST_LAYER + 1
        qc["total_quantized_sublayers"] = f"{num_layers * 7} ({num_layers} layers × 7 sublayers: q/k/v/o + gate/up/down)"
        qc["note"] = "L8-L32 chosen to preserve early layers (L0-L7) and late layers (L33-L35) in bf16. MLP + Attention all quantized."
        with open(qc_path, "w") as f:
            json.dump(qc, f, indent=2)
        qm.logger.info("[保存] quant_config.json 已更新 (2x16-AttnMLP-L8L32)")


# ═══════════════════════════════════════════════════════════════
# Monkey-patch: 确保 main() 使用 quantize.py 的覆写函数
# ═══════════════════════════════════════════════════════════════

qm.resolve_target_layers = resolve_target_layers
qm.update_outs = update_outs
qm.quantize_single_layer = quantize_single_layer
qm.verify_quantization_integrity = verify_quantization_integrity
qm.prepare_calibration_data = prepare_calibration_data
qm.save_quantized_model = save_quantized_model

# ═══════════════════════════════════════════════════════════════
# 启动 — GPU0
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    num_layers = TARGET_LAST_LAYER - TARGET_FIRST_LAYER + 1
    num_sublayers = num_layers * 7

    qm.logger.info("=" * 60)
    qm.logger.info(" RiverOne AQLM 2×16 量化 — L8-L32 (25层) MLP + Attention")
    qm.logger.info(f" scheme: 2×16 (2 codebooks, nbits=16, in_group=16)")
    qm.logger.info(f" codebook_size=65536, 等效位宽 ~1 bit/param")
    qm.logger.info(f" 量化范围: L{TARGET_FIRST_LAYER}-L{TARGET_LAST_LAYER} (共 {num_layers} 层)")
    qm.logger.info(f" ★ MLP + Attention — q/k/v/o + gate/up/down 全部量化")
    qm.logger.info(f" ★ 保留 L0-L7, L33-L35 为 bf16")
    qm.logger.info(f" ★ FAISS K-Means: k-means++ init, {qm.INIT_MAX_ITER} iter")
    qm.logger.info(f" 量化子层数: {num_layers}×7={num_sublayers} 个")
    qm.logger.info(f" 运行 GPU: cuda:0")
    qm.logger.info(f" 源模型: {qm.SOURCE_MODEL_PATH}")
    qm.logger.info(f" 输出目录: {qm.OUTPUT_DIR}")
    qm.logger.info(f" 日志文件: {qm.LOG_FILE}")
    qm.logger.info("=" * 60)

    qm.main()
