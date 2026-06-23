# RiverOne-QC-4B-v1-AQLM-miniViT PV-Tuning

## 概述

本目录对经过 **AQLM 量化 + miniViT 压缩** 的 `RiverOne-QC-4B-v1` 模型
执行 **PV-Tuning**（论文：[PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852)），
在 QcalEval SFT 数据上进一步恢复压缩损失的精度。

### 背景

RiverOne-QC-4B 是一个多模态大模型（InternVL3_5-4B LLM + Ising ViT 视觉编码器），
经 AQLM 1×16 方案量化为约 **1 bit/param**，再经 miniViT 蒸馏压缩视觉部分。
PV-Tuning 在此极端压缩的起点上，通过 **P/V 两步交替优化** 精调量化的 codebooks/scales/codes。

### PV-Tuning 方法概要

| 步骤 | 说明 |
|---|---|
| **P step**（连续优化） | 冻结 code 分配，通过反向传播更新 AQLM 的连续参数 `codebooks` 和 `scales` |
| **V step**（离散优化） | 冻结连续参数，用 AQLM L2 beam search 在小比例子空间中更新离散 `codes` |
| **子空间技巧** | 每次仅更新梯度最大的 top-τ code groups，保证步长足够大以跨越离散间隙 |
| **目标函数** | QcalEval JSONL 中 assistant token 的监督 cross-entropy 损失 |

与 STE（Straight-Through Estimator）不同，PV-Tuning 有收敛性保证（定理 3.1），
且在 1-2 bit 极端量化场景下显著优于纯 STE 方案。

## 目录结构

```
PV-tuning/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── prepare_data.sh              # 数据准备与检查脚本
├── validate_qcaleval.py         # JSONL + 图片引用校验
├── train_pv_tuning.py           # PV-Tuning 训练主脚本
├── run_pv_tuning.sh             # 一键启动训练
├── evaluate_perplexity.py       # 训练前后 PPL 对比评估
├── QcalEval/                    # 训练数据集
│   ├── qcaleval_zs_sft.jsonl    # 零样本 SFT 数据
│   ├── qcaleval_icl_sft.jsonl   # 上下文学习 SFT 数据
│   └── images/                  # 引用的图片文件（需自行放置）
└── outputs/                     # 训练输出（自动创建）
    └── pv_tuned_qcaleval/       # PV-tuned 模型
```

## 快速开始

### 1. 环境安装

```bash
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/PV-tuning
pip install -r requirements.txt
```

核心依赖：`torch`, `torchvision`, `transformers`, `safetensors`, `tqdm`, `pillow`

### 2. 数据准备

QcalEval JSONL 文件已就位，但图片需要自行下载/复制：

```bash
# 检查数据状态
bash prepare_data.sh

# 将图片文件复制到 QcalEval/images/
cp /path/to/qcaleval/images/*.png QcalEval/images/

# 校验数据完整性
python3 validate_qcaleval.py
```

### 3. 启动训练

```bash
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/PV-tuning
bash run_pv_tuning.sh
```

#### 常用配置覆盖

```bash
# 单 epoch 训练
EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=8 bash run_pv_tuning.sh

# 同时微调非量化参数（embedding、norm、mlp1 等）
UPDATE_NON_QUANTIZED=1 bash run_pv_tuning.sh

# 冒烟测试：10 步后保存退出
ALLOW_MISSING_IMAGES=1 DRY_RUN_STEPS=10 bash run_pv_tuning.sh

# 指定 GPU
DEVICE=cuda:1 bash run_pv_tuning.sh

# 自定义学习率
LR=1e-4 CODE_LR=5e-3 bash run_pv_tuning.sh

# 仅训练前 4 层（调试用）
MAX_QUANTIZED_LAYERS=4 bash run_pv_tuning.sh
```

### 4. 评估对比

训练前后分别评估 perplexity，量化 PV-Tuning 提升：

```bash
# 评估原始压缩模型（PV-Tuning 前）
python3 evaluate_perplexity.py \
  --model_dir ../miniViT/miniViT_distilled \
  --output_json eval_results/before_pv.json

# 评估 PV-Tuned 模型（训练后）
python3 evaluate_perplexity.py \
  --model_dir outputs/pv_tuned_qcaleval \
  --output_json eval_results/after_pv.json
```

## 关键参数详解

### 训练控制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 1 | 训练轮数，增大可进一步提升精度 |
| `--batch_size` | 1 | 每 GPU batch size |
| `--gradient_accumulation_steps` | 8 | 梯度累积步数，有效 batch = batch_size × grad_accum |
| `--max_length` | 4096 | 序列最大长度 |
| `--lr` | 3e-4 | P step 连续参数（codebooks/scales）学习率 |
| `--code_lr` | 1e-2 | V step proxy buffer 学习率（影响 code 更新幅度） |
| `--weight_decay` | 0.0 | AdamW 权重衰减 |

### V step（离散 code 更新）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--update_codes` | True | 是否更新离散 codes |
| `--max_code_change_per_step` | 1e-3 | 每步最多更新 0.1% 的 code groups |
| `--beam_size` | 1 | 离散搜索 beam 宽度（1x16 方案 beam=1 足够） |
| `--code_update_every` | 1 | 每隔 N 步更新一次 codes |
| `--delta_decay` | 0.0 | Proxy-to-quantized weight 衰减系数（0 表示不做 decay） |
| `--code_trust_ratio` | None | Trust region 限制（None = 不做限制） |

### 训练范围

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--update_codebooks_and_scales` | True | 是否更新连续量化参数 |
| `--update_non_quantized_parameters` | False | 是否同时微调非量化层 |
| `--freeze_vision` | True | 是否冻结视觉编码器 |
| `--max_quantized_layers` | None | 限制替换的 AQLM 层数（None = 全部 36 层） |

### 调试/冒烟

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dry_run_steps` | 0 | 运行 N 步后保存退出（0 = 正常训练） |
| `--save_every_steps` | 0 | 每 N 步保存 checkpoint（0 = 不保存中间点） |
| `--log_every_steps` | 1 | 每 N 步打印日志 |
| `--allow_missing_images` | False | 允许图片缺失（用灰色占位图替代） |

## 输出模型

PV-Tuning 完成后，模型保存到 `outputs/pv_tuned_qcaleval/`，目录结构与原始模型相同：

```
outputs/pv_tuned_qcaleval/
├── config.json                  # 模型配置
├── model.safetensors            # 所有权重（含更新后的 codebooks/scales/codes）
├── model.safetensors.index.json # 权重索引
├── tokenizer_config.json        # Tokenizer 配置
├── ...                          # 其他配置文件
```

PV-Tuned 模型保持与原始模型相同的 AQLM 权重结构（`.codebooks`, `.codes`, `.scales`），
可继续使用现有 AQLM 推理库加载。

## 模型架构说明

```
RiverOne-QC-4B-v1-AQLM-miniViT
│
├── vision_model (IsingVisionEncoder)   ← freeze_vision=True 时冻结
│   └── 27 层 IsingBlock
│
├── mlp1 (视觉→语言投影层)              ← update_non_quantized=True 时训练
│   └── Linear(2048→2560) + GELU + Linear(2560→2560)
│
└── language_model (Qwen3ForCausalLM)   ← 36 层 Decoder
    ├── embed_tokens                     ← 非量化，update_non_quantized=True 时训练
    ├── layers.0-35
    │   ├── input_layernorm              ← 非量化
    │   ├── self_attn
    │   │   ├── q_proj (AQLM 1x16)       ← PV-Tuning 更新 codebooks/scales/codes
    │   │   ├── k_proj (AQLM 1x16)
    │   │   ├── v_proj (AQLM 1x16)
    │   │   ├── o_proj (AQLM 1x16)
    │   │   └── q_norm, k_norm           ← 非量化
    │   ├── post_attention_layernorm     ← 非量化
    │   └── mlp
    │       ├── gate_proj (AQLM 1x16)    ← PV-Tuning 更新
    │       ├── up_proj (AQLM 1x16)
    │       └── down_proj (AQLM 1x16)
    ├── norm                             ← 非量化
    └── lm_head                          ← 非量化
```

每层 7 个 AQLM 线性投影（4 个 attention + 3 个 MLP），共 36 × 7 = 252 个 AQLM 层。

## AQLM 量化方案

- **方案**: 1×16（1 个 codebook，out_group_size=1，in_group_size=16）
- **码本大小**: codebook_size=65536（每个 code 16 bit）
- **等效位宽**: ~1 bit/param（每组 16 个权重用 1 个 16-bit code 索引）
- **code 数据类型**: torch.int16
- **codebook/scale 数据类型**: torch.bfloat16

## 注意事项

1. **图片数据**：正式训练必须提供真实 QcalEval 图片。`--allow_missing_images` 仅用于脚本链路冒烟测试。
2. **显存**：当前配置在单张 24GB GPU（RTX 3090/4090）上可运行压缩后的模型（~1.72B 存储元素，~3.20 GB）。如 OOM，降低 `--max_length` 或增大 `--gradient_accumulation_steps`。
3. **训练时间**：单 epoch 约需数小时（取决于 GPU 和数据量）。建议先 `DRY_RUN_STEPS=10` 验证流程。
4. **非量化参数**：默认 `--update_non_quantized_parameters=False` 仅更新 AQLM 参数。开启后会微调 embedding/lm_head/norm/mlp1 等非量化权重，可能进一步提升精度但需更多显存。
5. **视觉编码器**：默认 `--freeze_vision=True` 冻结 vision_model，因为 miniViT 已蒸馏压缩，不建议进一步微调。

## 参考

- [PV-Tuning 论文](https://arxiv.org/abs/2405.14852)
- [AQLM 论文](https://arxiv.org/abs/2401.06118)
- [InternVL 3.5](https://github.com/OpenGVLab/InternVL)
- [Qwen3](https://github.com/QwenLM/Qwen3)
