#!/usr/bin/env python3
"""
=============================================================================
 RiverOne-QC-4B-v2 AQLM 量化 — 2×16 scheme, 后8层, Attn(n14)+MLP(n16)
 校准数据: qcaleval_zs_sft.jsonl (多模态：图像→IsingViT→LLM hidden states)

 方案定义:
   num_codebooks=2, nbits_per_codebook=16, in_group_size=16, out_group_size=1
   等效位宽: ~3.35 bit/param (MLP)
   ★ 量化 Attn(n14)+MLP(n16) (gate_proj, up_proj, down_proj) — Attention 保持 bf16
   量化范围: L28-L35 (最后8层)
   
 ★ 此为持久化基准配置 — 不要随意修改参数 ★
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
# 配置
# ═══════════════════════════════════════════════════════════════

qm.SOURCE_MODEL_PATH = "/home/lxy/workspace/riverone-release/RiverOne-QC-4B-v2"
qm.OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "RiverOne-QC-4B-v2-AQLM-2x16-8L-Attn14MLP16")
qm.LOG_FILE = os.path.join(SCRIPT_DIR, "quantize_2x16_8L_attn14_mlp16.log")

qm.NUM_LAST_LAYERS = 8
qm.NUM_CODEBOOKS = 2
qm.NBITS_PER_CODEBOOK = 16
qm.NSAMPLES = 64
qm.MODEL_SEQLEN = 2048
qm.OFFLOAD_ACTIVATIONS = False
# ★ FAISS K-Means (k-means++ init + GPU 加速) + 增加迭代
qm.USE_FAISS = True
qm.INIT_MAX_ITER = 100
qm.INIT_MAX_POINTS_PER_CENTROID = 5
# ★ 量化 Attn(n14)+MLP(n16) 层 — Attention 保持 bf16
qm.LINEAR_LAYER_KEYWORDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
qm.SUBlayer_NBITS = {"k_proj": 14, "v_proj": 14}  # k/v nbits=14(n16384), rest nbits=16(n65536)

CALIBRATION_JSONL = "/home/lxy/workspace/datasets/vqa_format/qcaleval_zs_sft.jsonl"
IMAGE_BASE = "/home/lxy/workspace/datasets/vqa_format"

qm.logger = qm.setup_logging(qm.LOG_FILE)


# ═══════════════════════════════════════════════════════════════
# GPU-safe update_outs（同上个版本）
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
# ★ 多模态校准数据收集 ★
# ═══════════════════════════════════════════════════════════════

class _CatcherExit(Exception):
    pass

class _Catcher(nn.Module):
    """钩子模块：捕获 LLM 第一层输入后提前退出。"""
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
    """
    多模态校准数据收集：对每个 JSONL 条目加载图像，经过 IsingViT + mlp1 投影，
    捕获 LLM 第一层的多模态 hidden states。
    """
    from transformers import AutoProcessor

    qm.logger.info(f"[校准数据] 多模态模式: 图像→IsingViT→LLM hidden states")
    qm.logger.info(f"[校准数据] JSONL: {CALIBRATION_JSONL}")
    qm.logger.info(f"[校准数据] 图像目录: {IMAGE_BASE}")

    device = devices[0]
    
    # 加载 processor（处理图像+文本）
    proc = AutoProcessor.from_pretrained(
        qm.SOURCE_MODEL_PATH, trust_remote_code=True
    )
    
    # 读取 JSONL 条目
    entries = []
    with open(CALIBRATION_JSONL, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    qm.logger.info(f"[校准数据] 共 {len(entries)} 条 JSONL 条目")

    # 获取 LLM layers 和 hidden_size
    layers = qm.get_layers(model)
    hidden_size = qm.get_hidden_size(model)
    layer_device_original = next(layers[0].parameters()).device
    model_dtype = next(model.parameters()).dtype

    # 将第一层移到 GPU 并安装 Catcher
    layers[0] = layers[0].to(device)
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    
    # ★ 整个模型移到 GPU 以支持多模态前向（embedding 等必须在同设备）
    model = model.to(device)

    qm.logger.info(f"[校准数据] 开始逐条处理多模态样本...")
    qm.logger.info(f"[校准数据] 目标: {nsamples} 个样本 × {seqlen} tokens")
    
    inps_list = []  # 收集的 hidden states
    img_load_errors = 0
    skipped = 0
    
    from tqdm import tqdm
    
    for entry in tqdm(entries, desc="多模态校准"):
        if len(inps_list) >= nsamples:
            break
        
        # ★ 先提取文本（后续图像加载需要统计 <image> token 数量）
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
        
        # 加载图像 — 匹配 <image> token 数量
        img_field = entry.get("image", "")
        if isinstance(img_field, (list, tuple)):
            images_raw = img_field
        else:
            images_raw = [img_field]
        
        image_token_count = text.count("<image>")
        if image_token_count == 0:
            skipped += 1
            continue
        
        # ★ 替换 <image> → 处理器期望的 image_token (如 <IMG_CONTEXT>)
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
        
        # 处理多模态输入
        try:
            if len(images) == 1:
                images = images[0]  # 单图传 PIL Image 而非列表
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
        
        # 前向传播 — 捕获 LLM 第一层输入
        try:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )
        except _CatcherExit:
            hidden = catcher.captured  # [1, actual_seq_len, hidden_size]
            
            actual_len = hidden.shape[1]
            # 截断或填充到 seqlen
            if actual_len >= seqlen:
                hidden = hidden[:, :seqlen, :]
            else:
                pad = torch.zeros(
                    1, seqlen - actual_len, hidden_size,
                    dtype=hidden.dtype, device=device
                )
                hidden = torch.cat([hidden, pad], dim=1)
            
            inps_list.append(hidden[0])  # [seqlen, hidden_size]
        except Exception as e:
            skipped += 1
            continue
    
    # 恢复原始层
    layers[0] = catcher.module
    layers[0] = layers[0].to(layer_device_original)
    
    qm.logger.info(
        f"[校准数据] 收集完成: {len(inps_list)}/{nsamples} 样本, "
        f"图像加载失败: {img_load_errors}, 跳过: {skipped}"
    )
    
    if len(inps_list) < nsamples:
        qm.logger.warning(
            f"[校准数据] 样本不足 ({len(inps_list)} < {nsamples}), "
            f"将重复使用已有样本"
        )
        while len(inps_list) < nsamples:
            inps_list.append(inps_list[len(inps_list) % max(1, len(inps_list))])
    
    inps_list = inps_list[:nsamples]
    
    # 组织为 AQLM 期望的格式: List[Tensor] (每设备一个张量)
    # shape: [nsamples_per_device, seqlen, hidden_size]
    nsamples_per_device = (nsamples - 1) // len(devices) + 1
    inps = []
    for d in range(len(devices)):
        start = d * nsamples_per_device
        end = min(start + nsamples_per_device, nsamples)
        batch = torch.stack(inps_list[start:end])  # [n, seqlen, hidden]
        batch = batch.to(device=devices[d] if not offload_activations else "cpu")
        inps.append(batch)
    
    # 构建 forward_args
    rotary_emb = qm.get_rotary_emb_module(model)
    default_pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    forward_args = {
        "rotary_emb": rotary_emb,
        "default_position_ids": default_pos_ids,
    }
    
    qm.logger.info(f"[校准数据] 最终 inps: {[t.shape for t in inps]}")
    return inps, forward_args


# ═══════════════════════════════════════════════════════════════
# 自定义 prepare_calibration_data — 调用多模态收集
# ═══════════════════════════════════════════════════════════════

def _multimodal_prepare_calibration_data(
    model, tokenizer, nsamples, seqlen, dataset_name, seed, devices, offload_activations,
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
        qc["scheme"] = "2x16-Attn14MLP16-8L"
        qc["num_codebooks"] = 2
        qc["nbits_per_codebook"] = 16
        qc["in_group_size"] = 16
        qc["out_group_size"] = 1
        qc["quantized_layers"] = "last_8_of_36_llm_attn14_mlp16"
        qc["quantized_components"] = "attn_qkv_o + mlp_gate_up_down"
        qc["preserved_components"] = "attention_qkv_o, layers_0..27, miniViT, embedding, lm_head, norms"
        qc["source_model"] = "RiverOne-QC-4B-v2 (miniViT)"
        qc["kmeans"] = "FAISS k-means++, 100 iter, max_points_per_centroid=5"
        qc["calibration_data"] = "qcaleval_zs_sft.jsonl (multimodal: image→IsingViT→LLM)"
        qc["init_max_points_per_centroid"] = 5
        qc["sublayer_nbits"] = {"k_proj": 14, "v_proj": 14}
        with open(qc_path, "w") as f:
            json.dump(qc, f, indent=2)
        qm.logger.info("[保存] quant_config.json 已更新 (2x16-MLPonly)")

qm.save_quantized_model = _patched_save


# ═══════════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    qm.logger.info("=" * 60)
    qm.logger.info(" AQLM 2×16 量化 — 后8层 Attn14+MLP16 (FAISS k-means++)")
    qm.logger.info(f" scheme: 2×16 (2 codebooks, nbits=16, in_group=16)")
    qm.logger.info(f" codebook_size=65536, 等效位宽 ~3.4 bit/param (MLP)")
    qm.logger.info(f" ★ FAISS K-Means: k-means++ init, {qm.INIT_MAX_ITER} iter, max_points/centroid={qm.INIT_MAX_POINTS_PER_CENTROID}")
    qm.logger.info(f" ★ Attention (q/k/v/o_proj) 不量化 — 保持 bf16")
    qm.logger.info(f" 量化层数: 3 (gate/up/down) x 8 = 56 层")
    qm.logger.info(f" 校准模式: 图像→IsingViT(不压缩)→LLM hidden states")
    qm.logger.info(f" 源模型: {qm.SOURCE_MODEL_PATH}")
    qm.logger.info(f" 输出目录: {qm.OUTPUT_DIR}")
    qm.logger.info(f" 校准数据: {CALIBRATION_JSONL}")
    qm.logger.info(f" 图像目录: {IMAGE_BASE}")
    qm.logger.info("=" * 60)

    qm.main()
