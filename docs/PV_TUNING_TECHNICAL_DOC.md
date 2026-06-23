# RiverOne-QC-4B-v1-AQLM-miniViT PV-Tuning 训练技术文档

> **版本**: v1.0  
> **日期**: 2026-06-15  
> **目标模型**: `miniViT/miniViT_distilled`（~1.72B 存储元素，safetensors ~3.20 GB）  
> **参考论文**: [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852)

---

## 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [模型架构全貌](#2-模型架构全貌)
3. [PV-Tuning 方法论](#3-pv-tuning-方法论)
4. [技术实现细节](#4-技术实现细节)
5. [训练管线](#5-训练管线)
6. [数据集](#6-数据集)
7. [评估体系](#7-评估体系)
8. [超参数与调优建议](#8-超参数与调优建议)
9. [预期结果与基准](#9-预期结果与基准)
10. [运行指南](#10-运行指南)
11. [参考文献](#11-参考文献)

---

## 1. 项目背景与动机

### 1.1 问题陈述

大型多模态语言模型（MLLM）在端侧部署时面临严峻的存储与计算瓶颈。以 RiverOne-QC-4B-v1 为例，其原始 bf16 精度约需 **8.9 GB** 显存，远超消费级设备的承载能力。为在资源受限环境中高效推理，经过 AQLM 量化（1×16, ~1 bit/param）和 MiniViT 蒸馏压缩后，模型存储降至 **~3.20 GB**（约 1.72B 存储元素），但精度损失需要通过微调恢复。

### 1.2 技术路线概览

本项目的压缩-恢复流程分三步：

```
原始 RiverOne-QC-4B-v1 (bf16, ~8.9GB)
        │
        ▼
┌──────────────────────────────┐
│ Step 1: AQLM 量化 (1×16)     │  ← 将 LLM 权重压缩至 ~1 bit/param
│ 36层 × 7投影 → 252 个AQLM层  │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Step 2: MiniViT 视觉压缩     │  ← 蒸馏压缩视觉编码器（27→26层共享）
│ 层23/24 共享 MSA + MLP       │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Step 3: PV-Tuning 精度恢复   │  ★ 本文档核心
│ P步(连续优化) + V步(离散优化) │
│ 在 QcalEval SFT 数据上微调   │
└──────────────────────────────┘
        │
        ▼
   miniViT_distilled + PV-Tuned
```

### 1.3 为什么选择 PV-Tuning

传统极端量化微调方案存在根本性缺陷：

| 方案 | 原理 | 缺陷 |
|---|---|---|
| **纯连续微调** | 仅更新 codebooks/scales，冻结 codes | 可训练参数量极少（<0.1%），精度恢复有限 |
| **STE (Straight-Through Estimator)** | 将量化器的梯度直通传递 | 梯度偏差大，极易震荡发散；小学习率不更新，大学习率崩溃 |
| **Stochastic Rounding** | 随机舍入离散 codes | 收敛慢，方差大，难以稳定训练 |

**PV-Tuning** 通过 **P/V 两步交替优化 + 子空间下降** 从根本上解决了上述问题：
- 有严格收敛性保证（Theorem 3.1）
- 每次更新足够大的 code group 比例以跨越离散化间隙
- 同时优化连续参数（codebooks/scales）与离散参数（codes）

---

## 2. 模型架构全貌

### 2.1 总体结构

```
RiverOne-QC-4B-v1-AQLM-miniViT
│
├── vision_model: IsingVisionEncoder
│   ├── 27 层 IsingBlock (hidden=1152, heads=16)
│   ├── MiniViT 压缩: 层23/24 共享 MSA+MLP
│   ├── dwconv + transform_norm 新增参数
│   └── 输出维度: 2048
│
├── mlp1: 多模态投影
│   └── Linear(2048→2560) + GELU + Linear(2560→2560)
│
└── language_model: Qwen3ForCausalLM
    ├── embed_tokens: [151936, 2560]
    ├── layers.0 ~ layers.35 (36层 Decoder)
    │   ├── input_layernorm (RMSNorm)
    │   ├── self_attn
    │   │   ├── q_proj: AQLM 1×16   ←┐
    │   │   ├── k_proj: AQLM 1×16   ←┤
    │   │   ├── v_proj: AQLM 1×16   ←┤ 7 个 AQLM
    │   │   ├── o_proj: AQLM 1×16   ←┤ 投影层
    │   │   ├── q_norm (RMSNorm)     │ 每层
    │   │   └── k_norm (RMSNorm)     │
    │   ├── post_attention_layernorm │
    │   └── mlp                     │
    │       ├── gate_proj: AQLM 1×16←┤
    │       ├── up_proj:   AQLM 1×16←┤
    │       └── down_proj: AQLM 1×16←┘
    ├── norm (RMSNorm)
    └── lm_head: [2560, 151936]
```

### 2.2 关键参数统计

> **说明**："存储元素"指 safetensors 中实际保存的张量元素总数（含 int16 codes 和 bf16 权重）。
> 原始 RiverOne-QC-4B-v1 约 4.47B 参数（bf16, ~8.9 GB），经 AQLM + MiniViT 压缩后降至 ~1.72B 元素（~3.20 GB）。

| 组件 | 存储元素 | 存储占用 | 量化状态 |
|---|---|---|---|
| Vision Encoder (Ising ViT) | ~431M | ~0.86 GB | 未量化（MiniViT 蒸馏） |
| mlp1 投影层 | ~17M | ~0.03 GB | 未量化 |
| LLM embed_tokens | ~389M | ~0.78 GB | 未量化 |
| LLM lm_head | ~389M | ~0.78 GB | 未量化 |
| LLM Norm 层 | ~0.2M | ~0.0004 GB | 未量化 |
| AQLM codebooks (×252) | ~264M | ~0.53 GB | **量化** (bf16) |
| AQLM codes (×252) | ~227M | ~0.45 GB | **量化** (int16) |
| AQLM scales (×252) | ~1.1M | ~0.002 GB | **量化** (bf16) |
| **总计** | **~1,719M** | **~3.44 GB** | — |

> safetensors 文件实际大小约 **3.20 GB**（比原始 bf16 的 ~8.9 GB 减少 64%）。
> 其中 LLM Linear 投影部分从原始 ~3.95B params (7.9 GB) 压缩到 ~492M 元素 (0.98 GB)，压缩比约 **8×**。

### 2.3 AQLM 量化方案详解

| 参数 | 值 | 说明 |
|---|---|---|
| `scheme` | 1×16 | 1个codebook，out_group_size=1，in_group_size=16 |
| `codebook_size` | 65,536 | 每个code从 $2^{16}=65536$ 个候选中选择 |
| `code dtype` | int16 | 每个code占16 bits |
| `等效位宽` | ~1 bit/param | 每16个权重共享1个16-bit code：$16/16=1$ |
| `codebook dtype` | bfloat16 | codebook 向量以 bf16 存储 |
| `scales` | per-output-group | 每组一个scale因子，bf16 |

**数学表示**（单层单投影）：

给定权重矩阵 $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$，将其划分为 $g_{out} \times g_{in}$ 个组，每组尺寸 $1 \times 16$。对于第 $(i,j)$ 组：

$$\hat{\mathbf{W}}_{ij} = s_i \cdot \mathbf{C}[k_{ij}], \quad k_{ij} \in \{0, \dots, 65535\}$$

其中 $\mathbf{C} \in \mathbb{R}^{65536 \times 16}$ 为共享码本，$s_i$ 为第 $i$ 个输出组的缩放因子，$k_{ij}$ 为第 $(i,j)$ 组的离散码索引。

**每层code数量**（以 down_proj [2560, 9728] 为例）：

$$n_{codes} = \lceil 2560/1 \rceil \times \lceil 9728/16 \rceil = 2560 \times 608 = 1,556,480$$

### 2.4 MiniViT 视觉压缩

MiniViT 对 27 层 Ising ViT 的第 23/24 层进行参数共享：

| 共享组件 | 独立组件 | 新增参数 |
|---|---|---|
| MSA (qkv + proj) | norm1 | attn_transform_F1 (16×16) |
| MLP (fc1 + fc2) | norm2 | attn_transform_F2 (16×16) |
| — | — | mlp_dwconv (C=1152, K=3, groups=1152) |
| — | — | mlp_transform_norm (LayerNorm) |

---

## 3. PV-Tuning 方法论

### 3.1 问题形式化

设 $\mathcal{R}_c^d \subset \mathbb{R}^d$ 为所有可用 $c$ 个不同值表示的 $d$ 维向量的集合（即量化权重的可行域）。目标：

$$\min_{\mathbf{x} \in \mathcal{R}_c^d} \phi(\mathbf{x})$$

其中 $\phi: \mathbb{R}^d \to \mathbb{R}$ 为可微损失函数（如 cross-entropy），且有下界。

对于 AQLM 1×16，$\mathbf{x}$ 对应所有权重的展平向量，$c=65536$，$d$ 为总权重数。

### 3.2 P/V 两步框架

#### 核心概念

定义 $\mathbf{x} \in \mathcal{R}_c^d$ 的两个属性：

- **P(x)**：$\mathbf{x}$ 诱导的划分（Partition）——哪些权重共享相同的值
- **V(x)**：$\mathbf{x}$ 中不同值的集合（Values）——权重可以取哪些值

对于 AQLM：
- **P 由 codes 决定**：codes 确定每个权重组使用哪个 codebook 条目
- **V 由 codebooks/scales 决定**：codebooks 和 scales 决定每个 code 对应的实际浮点值

#### P 步（连续优化，固定 Partition）

$$M_P(\mathbf{x}) = \arg\min_{\mathbf{y} \in \mathbb{R}^d} \{\phi(\mathbf{y}) : P(\mathbf{y}) \supseteq P(\mathbf{x})\}$$

- **含义**：固定 code 分配（codes 不变），优化连续参数（codebooks, scales）
- **实现**：标准的反向传播 + AdamW 优化器
- **维度**：仅有 $O(c)$ 个可优化变量

#### V 步（离散优化，固定 Values）

$$M_V(\mathbf{y}) = \arg\min_{\mathbf{x} \in \mathbb{R}^d} \{\phi(\mathbf{x}) : V(\mathbf{x}) \subseteq V(\mathbf{y})\}$$

- **含义**：固定 codebook 值，仅更新 codes 分配
- **挑战**：搜索空间大小 $|V(\mathbf{y})|^d \leq 65536^d$，指数级

### 3.3 PV 算法与收敛性

```
Algorithm 1: PV Algorithm
─────────────────────────────────────────
1:  x₀ ∈ R_c^d                    ← 初始量化权重
2:  for k = 0, 1, ... do
3:      yₖ = M_P(xₖ)             ← P step: 固定 P, 优化 V
4:      xₖ₊₁ = M_V(yₖ)           ← V step: 固定 V, 优化 P
5:  end for
```

**Theorem 3.1（收敛性）**：设 $\phi$ 有下界，$x_0 \in \mathcal{R}_c^d$。则：
1. $y_k \in \mathcal{R}_{\leq c}^d$, $x_k \in \mathcal{R}_{\leq c}^d$ 对所有 $k \geq 0$ 成立
2. $\phi(x_{k+1}) \leq \phi(y_k) \leq \phi(x_k)$ 对所有 $k \geq 0$ 成立
3. 序列 $\{\phi(x_k)\}_{k=0}^\infty$ 收敛

### 3.4 线性化 V 步

直接求解 $M_V$ 在计算上不可行。利用 $\phi$ 的 $L$-光滑性：

$$\phi(\mathbf{x}) \approx \tilde{\phi}_{\mathbf{y}}(\mathbf{x}) = \phi(\mathbf{y}) + \langle\nabla\phi(\mathbf{y}), \mathbf{x}-\mathbf{y}\rangle + \frac{L}{2}\|\mathbf{x}-\mathbf{y}\|^2$$

由此导出 **线性化 V 步**：

$$\mathbf{x}^+ = \arg\min_{\mathbf{x}} \left\{\|\mathbf{x} - (\mathbf{y} - \tfrac{1}{L}\nabla\phi(\mathbf{y}))\|^2 : V(\mathbf{x}) \subseteq V(\mathbf{y})\right\}$$

即将 V 步转化为：在每个权重位置，从固定码本 $V(\mathbf{y})$ 中选择与「梯度下降后的目标值」最接近的码本条目。

**Lemma 3.3（单调性）**：若 $\phi$ 在 $\mathcal{R}_{\leq c}^d$ 上 $L$-光滑，则 $\phi(\mathbf{x}^+) \leq \phi(\mathbf{y})$。

### 3.5 子空间 V 步（核心技术）

线性化 V 步的关键缺陷：当步长 $1/L$ 过小时，离散码本中最接近的条目可能仍是当前条目本身，导致算法卡住。

**解决方案——子空间下降**：每步仅更新 $\tau \ll d$ 个权重：

$$\mathcal{S}_k \subset [d], \quad |\mathcal{S}_k| = \tau$$

$$\mathbf{x}^+ = \arg\min_{\mathbf{x}} \left\{\|\mathbf{x} - (\mathbf{y} - \tfrac{1}{L_{\mathcal{S}_k}}Z_k(\nabla\phi(\mathbf{y})))\|^2 : V(\mathbf{x}) \subseteq V(\mathbf{y})\right\}$$

其中 $Z_k$ 仅保留 $\mathcal{S}_k$ 中坐标的梯度，$L_{\mathcal{S}_k} \ll L$ 使得有效步长足够大。

**子空间选择策略**（贪婪法）：
1. 对每个 code group 计算 $\| \text{Adam update} \|_2$
2. 选择 top-$\tau$ 个 group 构成 $\mathcal{S}_k$
3. 仅对这些 group 执行 beam search 更新 codes

**关键洞察**：$L_{\mathcal{S}_k}$ 通常比全局 $L$ 小一个数量级以上，使得可以用足够大的学习率执行有意义的离散更新。

### 3.6 PV-Tuning 完整算法

```
Algorithm 2: PV-Tuning (Optimization)
──────────────────────────────────────────
0:  x₀ (初始压缩权重), φ (损失函数), τ (子空间大小)
1:  for k = 0, ..., K-1 do
2:      // P step: 更新 V(x) —— 反向传播
3:      yₖ = argmin_y { φ(y) : P(y) ⊇ P(xₖ) }
4:        → 冻结 codes，AdamW 优化 codebooks + scales
5:
6:      // V step: 更新 P(x) —— 子空间离散搜索
7:      Sₖ = arg top-τ |Adam update of weight proxy|
8:      xₖ₊₁ = argmin_x { φ̂_{yₖ, Sₖ}(x) : V(x) ⊆ V(yₖ) }
9:        → L2 beam search 更新 codes on Sₖ
10: end for
```

---

## 4. 技术实现细节

### 4.1 训练时 AQLM 模块

`TrainableAQLMLinear` 类封装训练时的可微 AQLM 层：

```
TrainableAQLMLinear
├── codebooks:   nn.Parameter [1, 65536, 1, 16]  ← P步更新 (requires_grad=True)
├── scales:      nn.Parameter [2560, 1, 1, 1]    ← P步更新
├── codes:       nn.Parameter [2560, 608, 1]     ← V步更新 (requires_grad=False)
├── weight_proxy nn.Parameter [2560, 9728]       ← STE buffer (requires_grad=True)
└── bias:        nn.Parameter (optional)
```

**前向传播**：
```python
def forward(self, input):
    weight = dequantize(codes, codebooks, scales)  # AQLM 解码
    if self.weight_proxy is not None:
        weight = weight + (proxy - proxy.detach())  # STE 校正项
    return F.linear(input, weight, bias)
```

**V 步 code 更新**：
```python
def pv_update_codes_(self, beam_size, max_update_fraction, trust_ratio, delta_decay):
    reference = self.weight_proxy.detach()  # STE buffer 作为参考
    new_codes = beam_search_optimal_codes(
        reference_weight=reference,
        codebooks=self.codebooks.detach(),
        prev_codes=self.codes,
        scales=self.scales.detach(),
        beam_size=beam_size,
        max_update_fraction=max_update_fraction,
        trust_ratio=trust_ratio,
    )
    self.codes.copy_(new_codes)
    if delta_decay > 0:
        # Proxy → Quantized 渐进对齐
        quantized = self.dequantize()
        self.weight_proxy = (1-δ) * proxy + δ * quantized
```

### 4.2 L2 Beam Search（离散 code 搜索）

对于 AQLM 1×16 方案（单 codebook），使用 **贪心最近邻** 快速路径：

$$k_{ij}^{new} = \arg\min_{k \in \{0,\dots,65535\}} \| \mathbf{C}[k] - \mathbf{w}_{ij}^{ref}/s_i \|^2$$

复杂度：$O(n_{codes} \times 65536 \times 16)$，其中 $n_{codes}$ 为本次更新的 code 数量。

**多 codebook 情况**（备选，当前 1×16 不使用）：使用 beam search 依次搜索每个 codebook，每个位置保留 beam_size 个候选。

### 4.3 优化器配置

| 参数组 | 参数 | 学习率 | β₁ | β₂ | Weight Decay |
|---|---|---|---|---|---|
| 连续量化参数 | codebooks, scales | 3e-4 | 0.90 | 0.95 | 0.0 |
| 非量化参数 | embed, norm, mlp1, lm_head | 3e-4 | 0.90 | 0.95 | 0.0 |
| Proxy 参数 | weight_proxy | 1e-2 | 0.0 | 0.95 | 0.0 |

> **设计理由**：Proxy 参数使用 β₁=0（无动量），β₂=0.95，更高学习率——因为 proxy buffer 需要快速追踪梯度方向以生成有效的 code 更新目标。

### 4.4 梯度累积与 V 步调度

```
for micro_step in dataloader:
    loss = model(batch).loss
    (loss / grad_accum_steps).backward()
    
    if micro_step % grad_accum_steps == 0:
        optimizer.step()                    # P step: 更新 codebooks/scales
        if step % code_update_every == 0:
            pv_update_all_codes(model)      # V step: 更新 codes
        optimizer.zero_grad()
```

- **梯度复用**：P 步和 V 步共用同一次 backward 的梯度，节省计算
- **code 更新频率**：默认每步都更新（`code_update_every=1`）
- **max_code_change_per_step**：每步仅更新 0.1% 的 code groups（~1500/1.5M），保证更新质量

### 4.5 混合精度与显存优化

```
训练精度策略:
├── load_dtype:     bfloat16    ← 模型加载精度
├── master_dtype:   float32     ← 可训练参数存储精度
├── buffer_dtype:   bfloat16    ← proxy buffer 精度
├── amp_dtype:      bfloat16    ← 自动混合精度前向
└── gradient_checkpointing: ON  ← 重计算换显存
```

**显存估算**（压缩模型 ~3.20 GB，单 GPU 24GB）：

| 组件 | 显存占用 |
|---|---|
| 模型参数 (bf16 + int16 codes) | ~3.4 GB |
| 优化器状态 (fp32, AdamW, 仅可训练参数) | ~2-4 GB |
| 梯度 (fp32, 仅可训练参数) | ~1-2 GB |
| 激活值 (grad_ckpt) | ~2-4 GB |
| **总计** | **~10-14 GB** |

> 当前默认配置（batch_size=1, grad_accum=8, freeze_vision=True, 仅更新 AQLM 参数）可在 24GB GPU 上运行。

---

## 5. 训练管线

### 5.1 数据流

```
QcalEval JSONL
  │
  ├── 解析 conversation (user/assistant)
  ├── _resolve_image() → PIL Image
  ├── dynamic_preprocess() → 多 tile 裁剪 + 归一化
  │     ├── 448×448 tiles (基于宽高比动态选择)
  │     ├── 可选 thumbnail (全局缩略图)
  │     └── Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
  │
  ├── _expand_image_tokens() → 文本占位符展开
  │     └── "<image>" → "<img><IMG_CONTEXT>×N×T</img>"
  │           N = num_image_token = 196
  │           T = num_tiles
  │
  ├── tokenizer.apply_chat_template() → tokenize
  │     └── labels: prefix → -100, assistant → token ids
  │
  └── Collator: padding + image_flags 生成
```

### 5.2 图像预处理参数

| 参数 | 值 | 说明 |
|---|---|---|
| `image_size` | 448 | 单 tile 尺寸 |
| `min_tiles` | 1 | 最少 tile 数 |
| `max_tiles` | 12 | 最多 tile 数 |
| `use_thumbnail` | True | 多 tile 时附加全局缩略图 |
| `num_image_token` | 196 | 每个 tile 的视觉 token 数 |
| `normalize_mean` | (0.5, 0.5, 0.5) | Ising ViT 归一化均值 |
| `normalize_std` | (0.5, 0.5, 0.5) | Ising ViT 归一化标准差 |

### 5.3 训练循环伪代码

```
1.  model = AutoModel.from_pretrained(miniViT_distilled)
2.  tokenizer = AutoTokenizer.from_pretrained(miniViT_distilled)
3.  quantized_modules = replace_aqlm_layers_for_training(model)
        // 读取 safetensors 中的 codebooks/codes/scales
        // 替换 nn.Linear → TrainableAQLMLinear
4.  configure_training(model, quantized_modules)
        // 冻结非训练参数，分组设置 lr
5.  dataset = QcalEvalSFTDataset(...)
6.  dataloader = DataLoader(dataset, collate_fn=QcalEvalCollator)

7.  for epoch in range(epochs):
        for batch in dataloader:
            with autocast(bf16):
                outputs = model(pixel_values, input_ids, labels, ...)
                loss = outputs.loss
            (loss / grad_accum).backward()
            
            if step % grad_accum == 0:
                optimizer.step()              // P step
                pv_update_all_codes(model)    // V step
                optimizer.zero_grad()

8.  save_model(model, tokenizer, output_dir)
```

---

## 6. 数据集

### 6.1 QcalEval 数据集

| 属性 | 值 |
|---|---|
| 零样本 SFT | `qcaleval_zs_sft.jsonl`（1458 条） |
| 上下文学习 SFT | `qcaleval_icl_sft.jsonl`（708 条） |
| 总计 | 2166 条 |
| 图片引用 | 5195 个（309 张独立图片） |
| 任务类型 | 量子计算实验分析（DRAG 校准、GMM 判别等） |
| 每条格式 | `{id, experiment_type, question_type, image, conversations: [user, assistant]}` |

### 6.2 数据预处理注意事项

1. **图片必须就位**：309 张图片需放入 `QcalEval/images/`。若缺失，设置 `--allow_missing_images` 会用灰色 (128,128,128) 占位图替代，但 **不可用于正式训练**。
2. **conversation 格式**：每条恰好 2 轮（user → assistant），多轮数据需提前拆分。
3. **`<image>` 占位符**：文本中的 `<image>` 会被展开为 `num_image_token × num_tiles` 个 `<IMG_CONTEXT>` token。

---

## 7. 评估体系

### 7.1 Perplexity 评估

在 QcalEval 验证集上计算 assistant token 的困惑度：

$$\text{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} \ell_i\right)$$

其中 $\ell_i$ 为第 $i$ 个 assistant token 的 cross-entropy loss。

```bash
# 评估基线模型
python3 evaluate_perplexity.py --model_dir ../miniViT/miniViT_distilled

# 评估 PV-tuned 模型
python3 evaluate_perplexity.py --model_dir outputs/pv_tuned_qcaleval
```

### 7.2 下游任务评估（建议）

推荐使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 在标准基准上评估：

| 任务 | 类型 | 指标 |
|---|---|---|
| WikiText-2 | 语言建模 | PPL ↓ |
| C4 | 语言建模 | PPL ↓ |
| WinoGrande | 常识推理 | Accuracy ↑ |
| PiQA | 物理常识 | Accuracy ↑ |
| HellaSwag | 常识推理 | Accuracy ↑ |
| ARC-easy/challenge | 科学推理 | Accuracy ↑ |

### 7.3 评估脚本

```bash
# 完整的训练前后对比
python3 evaluate_perplexity.py --model_dir ../miniViT/miniViT_distilled --output_json eval/before.json
bash run_pv_tuning.sh
python3 evaluate_perplexity.py --model_dir outputs/pv_tuned_qcaleval --output_json eval/after.json
python3 -c "
import json
before = json.load(open('eval/before.json'))
after  = json.load(open('eval/after.json'))
delta  = before['perplexity'] - after['perplexity']
print(f'PPL improvement: {delta:.2f} ({(delta/before[\"perplexity\"])*100:.1f}%)')
"
```

---

## 8. 超参数与调优建议

### 8.1 核心超参数

| 超参数 | 默认值 | 范围 | 影响 |
|---|---|---|---|
| `--lr` | 3e-4 | 1e-5 ~ 1e-3 | P 步学习率。过大→发散，过小→收敛慢 |
| `--code_lr` | 1e-2 | 5e-3 ~ 5e-2 | V 步 proxy 学习率。影响 code 更新幅度 |
| `--max_code_change_per_step` | 1e-3 | 1e-4 ~ 5e-2 | 每步更新的 code group 比例。越小更新越保守 |
| `--epochs` | 1 | 1 ~ 5 | 训练轮数。增大通常提升精度但收益递减 |
| `--gradient_accumulation_steps` | 8 | 1 ~ 32 | 有效 batch size |
| `--beam_size` | 1 | 1 ~ 4 | V 步搜索宽度。1×16 方案 beam=1 即可 |
| `--delta_decay` | 0.0 | 0.0 ~ 0.1 | Proxy→Quantized 对齐速率 |
| `--code_trust_ratio` | None | 0.01 ~ 0.1 | Trust region 限制。None=不限制 |

### 8.2 调优策略

#### 场景 A：显存不足 (OOM)

```bash
# 降低序列长度 + 增大梯度累积
MAX_LENGTH=2048 GRAD_ACCUM=16 bash run_pv_tuning.sh

# 或仅训练部分层
MAX_QUANTIZED_LAYERS=12 bash run_pv_tuning.sh
```

#### 场景 B：V 步无更新

如果日志显示 `mean_code_change=0.0`，说明 code 未变化：

```bash
# 增大 code 学习率 → 更大的梯度步长
CODE_LR=5e-2 bash run_pv_tuning.sh

# 或增大每次更新比例
MAX_CODE_CHANGE_PER_STEP=5e-3 bash run_pv_tuning.sh
```

#### 场景 C：Loss 震荡/发散

```bash
# 降低学习率 + 减小 code 更新比例
LR=1e-4 MAX_CODE_CHANGE_PER_STEP=1e-4 bash run_pv_tuning.sh

# 启用 trust region
TRUST_RATIO=0.05 bash run_pv_tuning.sh
```

#### 场景 D：追求最高精度

```bash
# 解冻非量化参数 + 多 epoch + 更大 batch
EPOCHS=3 UPDATE_NON_QUANTIZED=1 GRAD_ACCUM=16 bash run_pv_tuning.sh
```

### 8.3 推荐配置矩阵

| 场景 | lr | code_lr | max_code_change | epochs | update_non_quant |
|---|---|---|---|---|---|
| 快速验证 | 3e-4 | 1e-2 | 1e-3 | 1 | False |
| 标准训练 | 3e-4 | 1e-2 | 1e-3 | 2 | False |
| 高精度 | 1e-4 | 5e-3 | 5e-4 | 3 | True |
| 极低资源 | 3e-4 | 1e-2 | 1e-3 | 1 | False |

---

## 9. 预期结果与基准

### 9.1 PV-Tuning 论文基准（参考）

基于 PV-Tuning 论文 Table 1，在 Llama 2 7B 上的结果：

| 方法 | GPTQ 2.14bit | VQ 1.58bit | AQLM 2.01bit |
|---|---|---|---|
| 无微调 | 3290 PPL | 20.26 PPL | 7.38 PPL |
| 仅连续参数 | 16.77 PPL | 8.17 PPL | 6.69 PPL |
| STE | 8.79 PPL | 7.76 PPL | 6.41 PPL |
| **PV-Tuning (子空间)** | **8.49 PPL** | **7.38 PPL** | **6.13 PPL** |
| **PV-Tuning + STE** | **8.43 PPL** | **7.32 PPL** | **5.90 PPL** |

> 注意：上述为 Llama 2 的参考数据，RiverOne-QC-4B 的绝对 PPL 会因模型和数据集不同而异。关键是 **相对提升幅度**——PV-Tuning 通常在极端量化场景（≤2 bit）下比纯 STE 方案提升 10-20%。

### 9.2 RiverOne-QC-4B 预期提升

基于 AQLM 1×16 方案的压缩程度（~1 bit/param），合理的预期：

| 指标 | 预期改善 |
|---|---|
| QcalEval PPL | 相对基线下降 15-30% |
| 零样本准确率 | 相对提升 3-8 个百分点 |
| 推理速度 | **无影响**（权重量化方案不变） |

### 9.3 收敛曲线预期

```
Loss
 │
 │  ╲
 │   ╲___
 │       ╲___              ← P步：平滑下降
 │           ╲╲╲            ← V步：阶梯式下降（code更新时）
 │              ╲___
 │                  ╲___    ← 渐近收敛
 └────────────────────────→ Steps
```

---

## 10. 运行指南

### 10.1 环境准备

```bash
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/PV-tuning
pip install -r requirements.txt
```

### 10.2 数据准备

```bash
# 检查数据状态（列出缺失图片）
bash prepare_data.sh

# 将图片放入指定目录
cp /path/to/qcaleval/images/*.png QcalEval/images/

# 验证数据完整性
python3 validate_qcaleval.py
```

### 10.3 启动训练

```bash
# 标准训练
bash run_pv_tuning.sh

# 自定义参数
EPOCHS=2 LR=3e-4 CODE_LR=1e-2 GRAD_ACCUM=8 bash run_pv_tuning.sh

# 仅训练前 4 层（快速验证）
MAX_QUANTIZED_LAYERS=4 DRY_RUN_STEPS=10 ALLOW_MISSING_IMAGES=1 bash run_pv_tuning.sh
```

### 10.4 评估结果

```bash
python3 evaluate_perplexity.py --model_dir outputs/pv_tuned_qcaleval
```

### 10.5 输出模型使用

PV-Tuned 模型保持标准 HuggingFace 格式，可用与原始模型相同的方式加载：

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "outputs/pv_tuned_qcaleval",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/pv_tuned_qcaleval",
    trust_remote_code=True,
)
```

---

## 11. 参考文献

1. **PV-Tuning**: Malinovskii, V., Mazur, D., Ilin, I., et al. "PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression." *arXiv:2405.14852*, 2024.
2. **AQLM**: Egiazarian, V., Panferov, A., Kuznedelev, D., et al. "Extreme Compression of Large Language Models via Additive Quantization." *arXiv:2401.06118*, 2024.
3. **QuIP#**: Tseng, A., Chee, J., Sun, Q., et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." *arXiv:2402.04396*, 2024.
4. **GPTQ**: Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *arXiv:2210.17323*, 2022.
5. **STE**: Bengio, Y., Léonard, N., Courville, A. "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." *arXiv:1308.3432*, 2013.
6. **Coordinate Descent**: Richtárik, P., Takáč, M. "Parallel Coordinate Descent Methods for Big Data Optimization." *Mathematical Programming*, 156(1-2):433–484, 2016.
7. **MiniViT**: Zhang, J., et al. "MiniViT: Compressing Vision Transformers with Weight Multiplexing." *CVPR*, 2024.
8. **Qwen3**: Qwen Team. "Qwen3 Technical Report." 2025.
9. **InternVL 3.5**: Chen, Z., et al. "InternVL 3.5: Expanding Performance Boundaries of Open-Source Multimodal Models." 2025.

---

> **文档维护**: 如有问题或改进建议，请提交至 RiverOne-QC-4B 项目仓库。
