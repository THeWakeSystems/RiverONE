# RiverOne-QC-4B-v1 定向 AQLM 量化项目（全部36层 LLM）

## 项目概述

本项目对 **RiverOne-QC-4B-v1**（miniViT 视觉编码器版本）多模态大模型的语言模型（LLM）分支
进行定向 AQLM（Additive Quantization of Language Models）量化，实现模型压缩与加速。

### 核心特性

| 特性 | 说明 |
|------|------|
| **源模型** | RiverOne-QC-4B-v1（miniViT + Qwen3-4B） |
| **量化算法** | AQLM（Vahe1994/AQLM 官方实现） |
| **量化方案** | **1×16 scheme**（in_group_size=16, out_group_size=1, num_codebooks=1, nbits_per_codebook=16） |
| **量化范围** | **LLM 全部 36 层**的 Attention + MLP 线性权重 |
| **保留精度** | miniViT 视觉编码器、mlp1 多模态投影层、Embedding、LM Head、所有 Norm 层保持 BF16 |

### 与 RiverOne-QC-4B-AQLM-L 的区别

| 项目 | RiverOne-QC-4B-AQLM-L | RiverOne-QC-4B-v1-AQLM-miniViT |
|------|----------------------|-------------------------------|
| 源模型 | RiverOne-QC-4B（InternVL ViT） | RiverOne-QC-4B-v1（miniViT） |
| 视觉编码器 | InternVisionModel（较大） | IsingVisionEncoder（miniViT，较小） |
| 量化层数 | 可选 4/8/12/16/20/28/36 | 全部 36 层 |
| 输出目录 | RiverOne-QC-4B-AQLM-36L | RiverOne-QC-4B-v1-AQLM-36L |

---

## 目录结构

```
RiverOne-QC-4B-v1-AQLM-miniViT/          ← 项目根目录
├── aqlm_lib/                              ← AQLM 量化引擎（本地副本）
│   ├── aq_engine.py                       #   单层量化引擎（Hessian + 量化训练）
│   ├── main.py                            #   量化主流程 & 并行工具
│   ├── data/wikitext2/                    #   校准数据集（WikiText-2）
│   └── src/                               #   核心算法模块
│       ├── aq.py                          #     QuantizedWeight / QuantizedLinear
│       ├── modelutils.py                  #     模型解析工具
│       ├── datautils.py                   #     数据加载工具
│       ├── kmeans.py                      #     K-Means 聚类
│       └── ...
├── RiverOne-QC-4B-v1-AQLM-36L/            ← 量化后模型输出目录（全部36层）
│   ├── model-00001-of-XXXXX.safetensors    #   权重分片
│   ├── model.safetensors.index.json        #   权重索引
│   ├── config.json                         #   模型配置
│   ├── quant_config.json                   #   量化配置记录
│   ├── tokenizer_config.json               #   分词器配置
│   ├── vocab.json / merges.txt             #   词表文件
│   ├── configuration_riverone_qc.py        #   模型配置类
│   ├── modeling_riverone_qc.py             #   模型定义
│   └── modeling_ising_vit.py               #   miniViT 视觉编码器定义
└── scripts/                                ← 脚本与文档
    ├── quantize.py                         #   量化主脚本（全部36层）
    ├── verify.py                           #   效果验证脚本
    ├── load_aqlm_quantized.py              #   量化模型加载器
    ├── requirements.txt                    #   Python 依赖清单
    └── README.md                           #   本说明文档
```

---

## 环境要求

### 硬件
- **GPU**: NVIDIA GPU，建议 ≥24GB 显存（A100/RTX 4090/RTX 6000 Ada 等）
  - 全部36层量化需要更多显存，建议 ≥48GB（A100-80G 或双卡）
- **内存**: ≥64GB 系统内存
- **存储**: ≥30GB 可用空间（源模型约 8GB + 量化输出 + 临时文件）

### 软件
- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **Python**: 3.10+
- **CUDA**: 12.1+（需与 PyTorch 版本匹配）

### 依赖安装

```bash
# 1. 创建虚拟环境（推荐）
python3 -m venv aqlm_env
source aqlm_env/bin/activate

# 2. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/scripts
pip install -r requirements.txt

# 4. AQLM 量化引擎已在本地 aqlm_lib/ 目录中，无需额外配置
```

---

## 使用说明

### 第一步：执行量化（全部36层 LLM）

```bash
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python quantize.py
```

**量化过程说明：**
1. 加载源模型 `/home/hyba/lyc/RiverOne-QC-4B-v1`（miniViT 版本）
2. 自动解析模型结构，定位 LLM 全部 36 层（第 0-35 层）
3. 加载 WikiText-2 校准数据（64 样本 × 2048 tokens）
4. 逐层执行 AQLM 1×16 量化训练（K-Means 初始化 → Adam 优化 → Beam Search）
5. 保存量化后模型权重及配置文件到 `../RiverOne-QC-4B-v1-AQLM-36L/`

**预计耗时：**
- 校准数据收集：约 5-10 分钟
- 每层量化训练：约 3-5 分钟（36 层共约 108-180 分钟）
- 模型保存：约 5-10 分钟
- **总计约 2-3.5 小时**（根据 GPU 性能）

> **⚠️ 注意**：全部 36 层量化耗时较长，建议在 `tmux` 或 `screen` 会话中运行：
> ```bash
> tmux new -s aqlm_quantize
> cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/scripts
> python quantize.py 2>&1 | tee quantize.log
> ```

### 第二步：验证量化效果

```bash
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/scripts
python verify.py
```

**验证内容：**
- 结构完整性检查（确认量化范围正确，miniViT 未被量化）
- 显存占用对比（原始 vs 量化）
- 推理速度对比（tokens/秒）
- 生成文本质量直观对比

### 第三步：加载量化模型进行推理

```python
from load_aqlm_quantized import load_quantized_model

model, tokenizer = load_quantized_model(
    "/home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/RiverOne-QC-4B-v1-AQLM-36L"
)

response, _ = model.chat(
    tokenizer=tokenizer,
    pixel_values=None,  # 纯文本推理
    question="请介绍一下人工智能。",
    generation_config={"max_new_tokens": 256},
)
print(response)
```

---

## 量化参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `NUM_LAST_LAYERS` | 36 | 量化的最后 N 层数（36 = 全部 LLM 层） |
| `IN_GROUP_SIZE` | 16 | AQLM 输入分组大小（1×16 方案） |
| `OUT_GROUP_SIZE` | 1 | AQLM 输出分组大小（1×16 方案） |
| `NUM_CODEBOOKS` | 1 | 码本数量 |
| `NBITS_PER_CODEBOOK` | 16 | 每个码本的比特数（→ codebook_size=65536） |
| `NSAMPLES` | 64 | 校准数据样本数 |
| `MODEL_SEQLEN` | 2048 | 校准序列长度 |
| `MAX_EPOCHS` | 5 | 量化优化最大 epoch 数 |
| `LR` | 1e-4 | 量化训练学习率 |
| `BEAM_SIZE` | 1 | Beam Search 宽度 |
| `DEVICES` | ["cuda:0"] | 层传播/训练使用的 GPU |
| `DTYPE` | "bfloat16" | 模型加载精度 |

> **⚠️ 重要约束：** `IN_GROUP_SIZE`, `OUT_GROUP_SIZE`, `NUM_CODEBOOKS`, `NBITS_PER_CODEBOOK` 为硬性约束，**不得修改**。

---

## 量化范围详解

### 被量化的组件（全部 36 层 LLM Transformer）

```
language_model.model.layers.0.*   ─┐
language_model.model.layers.1.*    │
...                                 ├── 全部36层，每层包含：
language_model.model.layers.34.*   │   self_attn.q_proj  (nn.Linear)
language_model.model.layers.35.*  ─┘   self_attn.k_proj  (nn.Linear)
                                       self_attn.v_proj  (nn.Linear)
                                       self_attn.o_proj  (nn.Linear)
                                       mlp.gate_proj     (nn.Linear)
                                       mlp.up_proj       (nn.Linear)
                                       mlp.down_proj     (nn.Linear)

共: 36 层 × 7 个线性层 = 252 个量化矩阵
```

### 保持原精度的组件

| 组件 | 路径 | 原因 |
|------|------|------|
| 视觉编码器（miniViT） | `vision_model.*` | 保持视觉特征提取精度 |
| 多模态投影 | `mlp1.*` | 保持跨模态对齐精度 |
| LLM Embedding | `language_model.model.embed_tokens` | 保持词表映射精度 |
| LLM LM Head | `language_model.lm_head` | 保持输出分布精度 |
| 所有 Norm 层 | `*layernorm*`, `*norm*` | Norm 层参数量小，不量化 |

---

## 注意事项

1. **显存管理**：量化过程需额外显存用于 Hessian 矩阵（XTX）和优化器状态。若显存不足，可减少 `NSAMPLES` 或 `MODEL_SEQLEN`。
2. **全部36层耗时**：量化全部 36 层预计需要 2-3.5 小时，建议在后台运行。
3. **路径硬约束**：源模型路径和输出路径为硬性约束，修改需同步更新所有脚本中的配置。
4. **AQLM 引擎**：量化引擎已包含在项目 `aqlm_lib/` 目录中，脚本自动引用，无需额外配置。
5. **随机种子**：已固定 `SEED=42` 确保可复现。修改种子可能导致量化结果差异。
6. **校准数据**：默认使用 WikiText-2。如需更高精度可切换为 C4 或 RedPajama（修改 `DATASET` 参数）。
7. **miniViT 保护**：视觉编码器（miniViT）和 mlp1 投影层不会被量化，验证脚本会专门检查此项。

---

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `CUDA out of memory` | 显存不足 | 减少 NSAMPLES/MODEL_SEQLEN，开启 OFFLOAD_ACTIVATIONS |
| `无法定位 LLM 分支` | 模型架构不匹配 | 检查 LLM_KEYWORDS 是否覆盖模型属性名 |
| `ModuleNotFoundError: aqlm` | aqlm 未安装 | `pip install aqlm` |
| `找不到 aq_engine 模块` | AQLM 引擎路径错误 | 确认 aqlm_lib/ 在项目根目录 |
| `量化后模型加载失败` | 权重格式不兼容 | 检查 aqlm 版本 ≥1.1.0 |
| `生成结果乱码` | 分词器不匹配 | 确认使用源模型的分词器 |
| `vision_model 被意外量化` | 量化范围配置错误 | 检查 verify.py 的结构检查输出 |

---

## 相关资源

- **AQLM 论文**: [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118)
- **AQLM 官方仓库**: [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM)
- **RiverOne-QC-4B-v1**: 内部模型，基于 Qwen3-4B + Ising Vision Encoder（miniViT）
- **参考项目**: `/home/hyba/lyc/RiverOne-QC-4B-AQLM-L`（原始 ViT 版本的量化实现）
