#!/usr/bin/env python3
"""
=============================================================================
 RiverONE-2B-ZS AQLM 量化 — 2×16 scheme, L12-L19, MLP only
 校准数据: qcaleval_zs_sft.jsonl (多模态：图像→IsingViT→LLM hidden states)

 方案定义:
   num_codebooks=2, nbits_per_codebook=16, in_group_size=16, out_group_size=1
   ★ 仅量化 MLP (gate_proj, up_proj, down_proj) — Attention 保持 bf16
   量化范围: L12-L19 (8层，非最后N层)

 模型架构: RiverONE-2B-ZS = Qwen3-1.7B (28层, hidden=2048, intermediate=6144)
           + IsingViT (27层, hidden=1152)
=============================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import List
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import quantize as qm

# ═══════════════════════════════════════════════════════════════
# ★ RiverONE QC 模型类型补丁 (model_type="riverone_qc")
# ═══════════════════════════════════════════════════════════════

_orig_get_llm_model = qm.get_llm_model
_orig_get_llm_config = qm.get_llm_config
_orig_get_layers = qm.get_layers
_orig_get_quantizer_key_prefix = qm.get_quantizer_key_prefix
_orig_get_hidden_size_fn = qm.get_hidden_size


def _patched_get_llm_model(model):
    if getattr(model.config, "model_type", None) == "riverone_qc":
        return model.language_model
    return _orig_get_llm_model(model)


def _patched_get_llm_config(model):
    if getattr(model.config, "model_type", None) == "riverone_qc":
        return model.config.llm_config
    return _orig_get_llm_config(model)


def _patched_get_layers(model):
    if getattr(model.config, "model_type", None) == "riverone_qc":
        return model.language_model.model.layers
    return _orig_get_layers(model)


def _patched_get_quantizer_key_prefix(model):
    if getattr(model.config, "model_type", None) == "riverone_qc":
        return "language_model.model.layers"
    return _orig_get_quantizer_key_prefix(model)


def _patched_get_hidden_size(model):
    if getattr(model.config, "model_type", None) == "riverone_qc":
        return model.config.llm_config.hidden_size
    return _orig_get_hidden_size_fn(model)


qm.get_llm_model = _patched_get_llm_model
qm.get_llm_config = _patched_get_llm_config
qm.get_layers = _patched_get_layers
qm.get_quantizer_key_prefix = _patched_get_quantizer_key_prefix
qm.get_hidden_size = _patched_get_hidden_size


# ═══════════════════════════════════════════════════════════════
# ★ 指定层量化补丁 — 替代"最后N层"逻辑，精确指定 L12-L19
# ═══════════════════════════════════════════════════════════════

TARGET_LAYER_INDICES = [12, 13, 14, 15, 16, 17, 18, 19]  # L12-L19 (0-indexed)

_orig_resolve_target_layers = qm.resolve_target_layers


def _patched_resolve_target_layers(model, num_last_layers):
    """重载 resolve_target_layers，用精确层号替换'最后N层'算法。"""
    loc_llm = qm.locate_language_model(model)
    all_layers = qm.get_llm_layers(model)
    total_layers = len(all_layers)

    # 校验目标层范围
    for idx in TARGET_LAYER_INDICES:
        if idx >= total_layers:
            raise ValueError(
                f"目标层 L{idx} 超出模型总层数 {total_layers}！"
            )

    key_prefix = qm.get_quantizer_key_prefix(model)
    index_to_name = {}
    for idx in TARGET_LAYER_INDICES:
        layer = all_layers[idx]
        for name, _ in layer.named_modules():
            if name and "." not in name:
                index_to_name[idx] = f"{key_prefix}.{idx}"
                break

    qm.logger.info(f"[模型解析] LLM 分支类型: {type(loc_llm).__name__}")
    qm.logger.info(
        f"[模型解析] LLM 配置: hidden_size={qm.get_llm_config(model).hidden_size}, "
        f"num_hidden_layers={total_layers}"
    )
    qm.logger.info(f"[模型解析] Transformer 总层数: {total_layers}")
    qm.logger.info(
        f"[模型解析] ★ 指定量化层: L{TARGET_LAYER_INDICES[0]}-"
        f"L{TARGET_LAYER_INDICES[-1]} ({len(TARGET_LAYER_INDICES)} 层)"
    )
    qm.logger.info(f"[模型解析] 量化器键前缀: '{key_prefix}'")

    return loc_llm, all_layers, TARGET_LAYER_INDICES, index_to_name


qm.resolve_target_layers = _patched_resolve_target_layers


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

qm.SOURCE_MODEL_PATH = "/home/lxy/workspace/RiverONE-2B-ZS"
qm.OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "RiverONE-2B-ZS-AQLM-2x16-8L-MLPonly"
)
qm.LOG_FILE = os.path.join(SCRIPT_DIR, "quantize_2x16_8L_riverone2b_zs.log")

qm.NUM_LAST_LAYERS = 8
qm.NUM_CODEBOOKS = 2
qm.NBITS_PER_CODEBOOK = 16
qm.NSAMPLES = 64
qm.MODEL_SEQLEN = 2048
qm.OFFLOAD_ACTIVATIONS = False
qm.USE_FAISS = True
qm.INIT_MAX_ITER = 100
qm.INIT_MAX_POINTS_PER_CENTROID = 5
qm.LINEAR_LAYER_KEYWORDS = ["gate_proj", "up_proj", "down_proj"]

CALIBRATION_JSONL = "/home/lxy/workspace/datasets/vqa_format/qcaleval_zs_sft.jsonl"
IMAGE_BASE = "/home/lxy/workspace/datasets/vqa_format"

if "language_model" not in qm.LLM_KEYWORDS:
    qm.LLM_KEYWORDS.insert(0, "language_model")

qm.logger = qm.setup_logging(qm.LOG_FILE)


# ═══════════════════════════════════════════════════════════════
# GPU-safe update_outs
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def _gpu_safe_update_outs(layer, inps, outs, **forward_args):
    gpu_device = torch.device("cuda:0")
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


qm.update_outs = _gpu_safe_update_outs


# ═══════════════════════════════════════════════════════════════
# 量化后保持 GPU
# ═══════════════════════════════════════════════════════════════

_orig_qsl = qm.quantize_single_layer


def _gpu_keep_qsl(layer, layer_idx, inps, outs, args, forward_args, model):
    result = _orig_qsl(layer, layer_idx, inps, outs, args, forward_args, model)
    return result.to(device=args.devices[0])


qm.quantize_single_layer = _gpu_keep_qsl


# ═══════════════════════════════════════════════════════════════
# ★ 多模态校准数据收集
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


def _multimodal_collect_layer_inputs(
    model, tokenizer, nsamples, seqlen, devices, offload_activations,
):
    from transformers import AutoProcessor

    qm.logger.info(f"[校准数据] 多模态模式: 图像→IsingViT→LLM hidden states")
    qm.logger.info(f"[校准数据] JSONL: {CALIBRATION_JSONL}")
    qm.logger.info(f"[校准数据] 图像目录: {IMAGE_BASE}")

    device = devices[0]

    proc = AutoProcessor.from_pretrained(
        qm.SOURCE_MODEL_PATH, trust_remote_code=True
    )

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
                text=text,
                images=images,
                return_tensors="pt",
                max_length=seqlen * 4,
                truncation=True,
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
                model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )
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


def _multimodal_prepare_calibration_data(
    model, tokenizer, nsamples, seqlen, dataset_name, seed, devices,
    offload_activations,
):
    return _multimodal_collect_layer_inputs(
        model, tokenizer, nsamples, seqlen, devices, offload_activations
    )


qm.prepare_calibration_data = _multimodal_prepare_calibration_data


# ═══════════════════════════════════════════════════════════════
# 修正 quant_config.json
# ═══════════════════════════════════════════════════════════════

_original_save = qm.save_quantized_model


def _patched_save(model, output_dir, source_dir):
    _original_save(model, output_dir, source_dir)
    qc_path = os.path.join(output_dir, "quant_config.json")
    if os.path.exists(qc_path):
        with open(qc_path, "r") as f:
            qc = json.load(f)
        qc["scheme"] = "2x16-MLPonly-8L"
        qc["num_codebooks"] = 2
        qc["nbits_per_codebook"] = 16
        qc["in_group_size"] = 16
        qc["out_group_size"] = 1
        qc["quantized_layers"] = "L12-L19 (8 layers of 28 MLP only)"
        qc["quantized_components"] = "mlp_gate_up_down"
        qc["preserved_components"] = (
            "attention_qkv_o, layers_0..11+20..27, IsingViT, embedding, lm_head, norms"
        )
        qc["source_model"] = "RiverONE-2B-ZS (Qwen3-1.7B + IsingViT)"
        qc["kmeans"] = "FAISS k-means++, 100 iter, max_points_per_centroid=5"
        qc["calibration_data"] = (
            "qcaleval_zs_sft.jsonl (multimodal: image→IsingViT→LLM)"
        )
        qc["init_max_points_per_centroid"] = 5
        qc["llm_config"] = {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "model_type": "qwen3",
        }
        with open(qc_path, "w") as f:
            json.dump(qc, f, indent=2)
        qm.logger.info("[保存] quant_config.json 已更新 (2x16-MLPonly-8L RiverONE-2B-ZS)")


qm.save_quantized_model = _patched_save


# ═══════════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    qm.logger.info("=" * 60)
    qm.logger.info(" AQLM 2×16 量化 — RiverONE-2B-ZS, L12-L19 MLP only")
    qm.logger.info(f" scheme: 2×16 (2 codebooks, nbits=16, in_group=16)")
    qm.logger.info(f" codebook_size=65536, 等效位宽 ~3.4 bit/param (MLP)")
    qm.logger.info(f" ★ FAISS K-Means: k-means++ init, 100 iter")
    qm.logger.info(f" ★ Attention (q/k/v/o_proj) 不量化 — 保持 bf16")
    qm.logger.info(f" 量化层数: 3 (gate/up/down) × 8 = 24 层")
    qm.logger.info(f" 量化范围: ★ L12-L19 (指定中间8层，非最后N层)")
    qm.logger.info(f" LLM: Qwen3-1.7B (hidden=2048, intermediate=6144)")
    qm.logger.info(f" 校准模式: 图像→IsingViT→LLM hidden states")
    qm.logger.info(f" 源模型: {qm.SOURCE_MODEL_PATH}")
    qm.logger.info(f" 输出目录: {qm.OUTPUT_DIR}")
    qm.logger.info(f" 校准数据: {CALIBRATION_JSONL}")
    qm.logger.info(f" 图像目录: {IMAGE_BASE}")
    qm.logger.info("=" * 60)

    qm.main()
