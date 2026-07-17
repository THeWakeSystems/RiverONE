# RiverOne 正式发布管线: AQLM 量化 → PV-Tuning 恢复 → MiniViT 共享知识蒸馏

> **正式发布配置名称**: `pv_fc_attnmlp_zs_sft_miniViT_21_24_distilled`
>
> **命名解码**:
> - `pv_fc` ― PV-Tuning with full codebook updates (codes 可更新，非冻结)
> - `attnmlp` ― 同时训练 Attention (q/k/v/o) 和 MLP (gate/up/down) 的 AQLM 层
> - `zs_sft` ― 仅使用零样本 SFT 数据 (QcalEval zs_sft, 1458 条)
> - `miniViT_21_24` ― MiniViT 双重共享: block 21→22 + block 23→24
> - `distilled` ― 知识蒸馏后模型 (Teacher→Student 软标签迁移)
>
> **实现途径**: quantize → finetune → compress → paramgen

---

## 管线总览

```
原始 RiverOne-QC-4B-v1 (bf16, ~8.9 GB, ~4.47B params)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: AQLM 量化 (quantize/)                               │
│ · 对 LLM 第 8-32 层 (25 层) 执行 2×16 加法量化              │
│ · 175 个线性投影矩阵 → 175 组 (codebooks + codes + scales)   │
│ · LLM: 7.9GB → 0.98GB, ~8× 压缩                             │
│ · 保留 L0-L7, L33-L35, ViT, embedding, lm_head, norms bf16  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: PV-Tuning (finetune/train_pv_tuning.py)             │
│ · P step: 反向传播更新 codebooks/scales                      │
│ · V step: L2 beam search 更新 top-τ subspace codes           │
│ · QcalEval ZS-SFT 数据, 3 epochs, 4 GPU 并行                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3a: MiniViT 第一对 (compress/apply_minivit_23_24.py)   │
│ · ViT block 23 → block 24 权重复用                           │
│ · MSA + MLP 共享, Norm 独立, +变换矩阵 (~12K params)         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3b: MiniViT 蒸馏第一对 (compress/distill_minivit_23_24.py) │
│ · Teacher (原始 ViT) → Student (共享权重 ViT)               │
│ · L_total = L_pred + L_attn + L_hddn                        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3c: MiniViT 第二对 (compress/apply_minivit_21_22.py)   │
│ · ViT block 21 → block 22 权重复用 (叠加在已蒸馏模型上)     │
│ · 变换矩阵 ~12K, 已有 block[24] 参数冻结                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3d: MiniViT 蒸馏第二对 (compress/distill_minivit_21_22.py) │
│ · 5 项损失: L_pred + L_attn + L_hddn + L_CE×3               │
│ · 新增 block[22] 参数可训练, block[24] 已有参数冻结            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
   pv_fc_attnmlp_zs_sft_miniViT_21_24_distilled (~3.20 GB, ~1.72B 存储元素)
```

---

## Stage 1: AQLM 量化 — 量子态离散化

### 1.1 量化方案

| 参数 | 值 |
|---|---|
| Scheme | **2×16** (2 codebooks, 各 16-bit) |
| in_group_size | 16 |
| out_group_size | 1 |
| num_codebooks | 2 |
| Codebook size | 65,536 (16-bit per code) |
| 等效位宽 | ~1 bit/param |
| Code dtype | int16 |
| Codebook/Scale dtype | bfloat16 |

### 1.2 量化范围

```
LLM 第 8-32 层 (language_model.model.layers.8 ~ 32, 共 25 层):
  每层 7 个线性投影:
    self_attn.q_proj  (AQLM 2×16)
    self_attn.k_proj  (AQLM 2×16)
    self_attn.v_proj  (AQLM 2×16)
    self_attn.o_proj  (AQLM 2×16)
    mlp.gate_proj     (AQLM 2×16)
    mlp.up_proj       (AQLM 2×16)
    mlp.down_proj     (AQLM 2×16)

共: 25 × 7 = 175 个 AQLM 量化矩阵
```

### 1.3 保持原精度的组件

| 组件 | 路径 | 原因 |
|---|---|---|
| LLM 第 0-7 层 | `language_model.model.layers.0 ~ 7` | 保留早期语义理解能力 |
| LLM 第 33-35 层 | `language_model.model.layers.33 ~ 35` | 保留晚期输出质量 |
| 视觉编码器 (miniViT) | `vision_model.*` | 保持视觉特征提取精度 |
| 多模态投影 | `mlp1.*` | 保持跨模态对齐精度 |
| LLM Embedding | `language_model.model.embed_tokens` | 保持词表映射 |
| LLM LM Head | `language_model.lm_head` | 保持输出分布 |
| 所有 Norm 层 | `*layernorm*`, `*norm*` | 参数量极小 |

### 1.4 量化训练参数

| 参数 | 值 |
|---|---|
| 校准数据 | WikiText-2 |
| 样本数 | 64 × 2048 tokens |
| 初始化 | K-Means 聚类 |
| 优化器 | Adam (lr=1e-4) |
| 最大 epoch | 5 |
| Beam size | 1 |
| 随机种子 | 42 |
| GPU | CUDA 设备 (≥24GB VRAM) |
| 预计耗时 | 2-3.5 小时 (A100) |

### 1.5 数学表示

对于权重矩阵 **W** ∈ ℝ^{d_out × d_in}，划分为 g_out × g_in 个组，每组 1×16:

```
Ŵ_{ij} = s_i · Σ_{m=1}^{2} C_m[k_{ij}^{(m)}],   k_{ij}^{(m)} ∈ {0, ..., 65535}
```

其中 **C_1, C_2** ∈ ℝ^{65536×16} 为两个共享码本，s_i 为输出组缩放因子，k_{ij}^{(1)}, k_{ij}^{(2)} 为两个离散码索引。通过两个码本向量的加性组合实现对每组 16 个权重的更精细建模。

### 1.6 压缩效果

| 指标 | 压缩前 | 压缩后 | 压缩比 |
|---|---|---|---|
| LLM Linear | ~3.95B params (7.9 GB) | ~492M elements (0.98 GB) | **8×** |
| 全模型 | ~4.47B params (8.9 GB) | ~1.72B elements (3.20 GB) | **2.8×** |

---

## Stage 2: PV-Tuning — 变分量子-经典优化

### 2.1 方法论

PV-Tuning 通过 **P/V 两步交替优化** 在极端量化场景下恢复精度:

| 步骤 | 操作 | 数学含义 |
|---|---|---|
| **P step** | 冻结 codes，AdamW 更新 codebooks/scales | min_{y} φ(y) s.t. P(y) ⊇ P(x_k) |
| **V step** | 冻结 codebooks/scales，L2 beam search 更新 top-τ codes | min_{x} φ̂_{y_k,S_k}(x) s.t. V(x) ⊆ V(y_k) |
| **子空间** | 仅更新梯度最大的 τ 组 codes | τ ≤ 0.1% total codes |

**收敛性保证** (Theorem 3.1): φ(x_{k+1}) ≤ φ(y_k) ≤ φ(x_k)，损失单调递减收敛。

### 2.2 TrainableAQLMLinear 模块

```python
class TrainableAQLMLinear(nn.Module):
    codebooks:   nn.Parameter [2, 65536, 1, 16]  ← P步更新 (requires_grad=True)
    scales:      nn.Parameter [d_out, 1, 1, 1]   ← P步更新 (requires_grad=True)
    codes:       nn.Parameter [d_out, d_in//16, 2] ← V步更新 (requires_grad=False)
    weight_proxy nn.Parameter [d_out, d_in]       ← STE buffer (requires_grad=True)
    bias:        nn.Parameter (optional)

def forward(self, input):
    weight = dequantize(codes, codebooks, scales)    # AQLM 解码
    weight = weight + (proxy - proxy.detach())       # STE 校正
    return F.linear(input, weight, bias)

def pv_update_codes_(self, beam_size, max_update_fraction, ...):
    reference = self.weight_proxy.detach()            # STE buffer 作为目标
    new_codes = beam_search_optimal_codes(            # L2 beam search
        reference_weight=reference,
        codebooks=self.codebooks.detach(),
        prev_codes=self.codes,
        scales=self.scales.detach(),
        beam_size=beam_size,
        max_update_fraction=max_update_fraction,       # top-τ 子空间
    )
    self.codes.copy_(new_codes)
```

### 2.3 训练超参数

#### 核心参数

| 参数 | 值 | 含义 |
|---|---|---|
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 1 | 每 GPU batch size |
| `--gradient_accumulation_steps` | 8 | 有效 batch = 8 |
| `--max_length` | 4096 | 序列最大长度 |
| `--lr` | 3e-4 | P step 连续参数学习率 |
| `--code_lr` | 1e-2 | V step proxy buffer 学习率 |
| `--max_code_change_per_step` | 1e-3 | 每步更新最多 0.1% codes |
| `--beam_size` | 1 | V step 搜索宽度 |
| `--code_update_every` | 1 | 每步都更新 codes |
| `--delta_decay` | 0.0 | Proxy→Quantized 对齐速率 |
| `--lr_scheduler` | cosine | 余弦退火 (10% warmup) |

#### 优化器分组

| 参数组 | 学习率 | β₁ | β₂ | Weight Decay |
|---|---|---|---|---|
| codebooks + scales (P step) | 3e-4 | 0.90 | 0.95 | 0.0 |
| weight_proxy (V step STE) | 1e-2 | 0.0 | 0.95 | 0.0 |
| 非量化参数 (可选) | 3e-4 | 0.90 | 0.95 | 0.0 |

> **设计理由**: Proxy 参数 β₁=0 (无动量) 确保快速追踪梯度方向; 高 lr=1e-2 保证 code 更新有足够步长跨越离散间隙。

#### 训练控制

| 参数 | 值 | 含义 |
|---|---|---|
| `--update_codes` | True | 执行 V step code 更新 |
| `--update_codebooks_and_scales` | True | 执行 P step 连续参数更新 |
| `--update_non_quantized_parameters` | False | 不微调 embed/norm/lm_head |
| `--freeze_vision` | True | 冻结 miniViT 视觉编码器 |
| `--gradient_checkpointing` | True | 激活重计算换显存 |

#### 精度策略

| 参数 | 值 |
|---|---|
| load_dtype | bfloat16 |
| master_dtype | float32 |
| buffer_dtype | bfloat16 |
| amp_dtype | bfloat16 |

### 2.4 多 GPU 并行

```
4 GPU 配置 (GPU 7,6,5,4):
  layers  0-8   → GPU 0 (cuda:0)
  layers  9-17  → GPU 1 (cuda:1)
  layers 18-26  → GPU 2 (cuda:2)
  layers 27-35  → GPU 3 (cuda:3)

最后层输出自动回传 GPU 0 用于 final norm + lm_head
```

### 2.5 训练数据

| 属性 | 值 |
|---|---|
| 零样本 SFT | `qcaleval_zs_sft.jsonl` (1458 条) |
| 上下文 SFT | `qcaleval_icl_sft.jsonl` (708 条, 本次未使用) |
| 图片引用 | 309 张独立图片 |
| 每条格式 | `{id, experiment_type, question_type, image, conversations: [user, assistant]}` |

**图像预处理**:
```
image_size=448, min_tiles=1, max_tiles=12, use_thumbnail=True
Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
num_image_token=196 per tile
```

### 2.6 训练循环

```python
model = AutoModel.from_pretrained(miniViT_distilled, device_map="cpu")
quantized_modules = replace_aqlm_layers_for_training(model)  # 175 层
configure_training(model, quantized_modules)                  # 367.8M 连续参数

for epoch in range(3):
    for micro_step, batch in enumerate(dataloader):
        with autocast(bf16):
            outputs = model(pixel_values, input_ids, labels, ...)
            loss = outputs.loss

        (loss / 8).backward()  # grad_accum=8

        if micro_step % 8 == 0:
            optimizer.step()                      # P step
            scheduler.step()                      # cosine LR
            pv_update_all_codes(quantized_modules) # V step
            optimizer.zero_grad()
            print(f"[PV] step={global_step} loss={avg_loss:.6f} mean_code_change={change:.8f}")
```

### 2.7 训练日志关键指标

从正式训练日志 (`pv_fc_attnmlp_zs_sft_20260627_0733.log`):

```
[PV] replaced 175 AQLM layers with training modules
[PV] splitting model across 4 GPUs (layers 0-8→GPU0, 9-17→GPU1, 18-26→GPU2, 27-35→GPU3)
[PV] trainable params: continuous=367,769,600, proxy=0, non_quantized=0
[PV] scheduler: cosine, warmup_steps=54, total_steps=546

# 训练过程中 loss 变化:
step=1   loss=0.170556  mean_code_change=0.0
step=2   loss=0.125010  mean_code_change=0.0
step=3   loss=0.111798  mean_code_change=0.0
step=4   loss=0.018900  mean_code_change=0.0
step=5   loss=0.095829  mean_code_change=0.0
...
```

---

## Stage 3: MiniViT 视觉压缩 — 纠缠驱动复用

### 3.1 配置总览

| 配置项 | 第一对 (23→24) | 第二对 (21→22) |
|---|---|---|
| 源块 (source) | 23 (倒数第4层) | 21 (倒数第6层) |
| 目标块 (target) | 24 (倒数第3层) | 22 (倒数第5层) |
| 共享组件 | MSA (qkv+proj) + MLP (fc1+fc2) | 同左 |
| 独立组件 | norm1, norm2 | norm1, norm2 |
| 变换矩阵 | F1(16×16), F2(16×16), dwconv, transform_norm | 同左 |
| 蒸馏状态 | 训练后冻结 | 可训练 (第一对已冻结) |

### 3.2 权重复用细节 (apply_minivit.py)

**Step 1**: 加载 AQLM 量化后的基模型

```python
model = AutoModel.from_pretrained(SOURCE_MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16)
# 通过 _load_aqlm() 将 LLM Linear 替换为 AQLM QuantizedLinear
```

**Step 2**: 权重复用

```python
# block target 共享 block source 的 MSA/MLP 权重
tgt_block.attn = src_block.attn  # MSA (qkv + proj) 共享
tgt_block.mlp  = src_block.mlp   # MLP (fc1 + fc2) 共享
# norm1, norm2 保持独立
```

**Step 3**: 添加变换矩阵 (~12K per pair)

```python
# Attention 变换矩阵: F1, F2 ∈ ℝ^{16×16} (初始化为单位矩阵)
tgt_block.register_parameter("attn_transform_F1_weight", nn.Parameter(torch.eye(16)))
tgt_block.register_parameter("attn_transform_F2_weight", nn.Parameter(torch.eye(16)))

# MLP 深度卷积变换: Conv1d(1152, 1152, 3, groups=1152)
dwconv = nn.Conv1d(1152, 1152, 3, padding=1, groups=1152)
nn.init.dirac_(dwconv.weight)  # Dirac 初始化 (近恒等)
tgt_block.register_module("mlp_dwconv", dwconv)

# MLP 变换归一化: LayerNorm(1152)
tgt_block.register_module("mlp_transform_norm", nn.LayerNorm(1152, eps=1e-6))
```

**Step 4**: 生成修改的 modeling_ising_vit.py + 保存权重

```python
# 在 IsingVisionEncoder.forward 中嵌入条件判断:
# if i == 22 or i == 24:
#     x = self._forward_block_24_shared(x)

# _forward_block_24_shared 方法:
#   x_normed = block_XX.norm1(x)
#   qkv = block_source.attn.qkv(x_normed)
#   scores = q @ k^T / sqrt(H)
#   scores = einsum("bmnl,mk->bknl", scores, F2)        # F2 变换
#   attn_weights = softmax(scores)
#   attn_out = einsum("bmnl,bklh->bmknh", attn, v)
#   attn_out = einsum("bmknh,km->bmknh", attn_out, F1)  # F1 变换
#   attn_out = block_source.attn.proj(attn_out_merged)
#   x = residual + attn_out
#
#   x_normed = block_XX.norm2(x)
#   x = dwconv(x_normed)                                # depthwise conv
#   x = transform_norm(x)                               # LayerNorm
#   x = block_source.mlp(x)                              # 共享 MLP
#   x = residual + x
```

### 3.3 知识蒸馏 (distill_minivit.py)

#### 第一对蒸馏 (23→24)

**损失函数** (MiniViT 论文 §3.3):

```
L_total = L_pred + β·L_attn + γ·L_hddn   (β=γ=1.0)

L_pred  = MSE(Teacher.merger_output, Student.merger_output)
L_attn  = MSE(Teacher.block[24].attn_weights, Student.block[24].attn_weights)
L_hddn  = MSE(Gram(Teacher.block[24].hidden), Gram(Student.block[24].hidden))

Gram(x) = (x / ||x||) @ (x / ||x||)^T
```

**可训练参数** (~12K):
```
block[24].attn_transform_F1_weight  (16×16 = 256)
block[24].attn_transform_F2_weight  (16×16 = 256)
block[24].mlp_dwconv.weight         (1152×3 = 3,456)
block[24].mlp_dwconv.bias           (1152)
block[24].mlp_transform_norm.weight (1152)
block[24].mlp_transform_norm.bias   (1152)
block[24].norm1.weight              (1152)
block[24].norm1.bias                (1152)
block[24].norm2.weight              (1152)
block[24].norm2.bias                (1152)
─────────────────────────────────────────
合计:                              ~12,228 params
```

**训练配置**:
| 参数 | 值 |
|---|---|
| 优化器 | Adam (lr=1e-3) |
| Batch size | 4 |
| Epochs | 10 |
| 训练数据 | 随机图像 (448×448, bf16) |
| 显存 | ≥24GB (同时加载 Teacher + Student) |

#### 第二对蒸馏 (21→22, 叠加)

**损失函数** (5 项):

```
L_total = L_pred + L_attn + L_hddn + L_CE-layer

L_CE-layer = α1·L_ce1  + α2·L_ce2  + α3·L_ce3
            (α1=0.7)     (α2=0.3)     (α3=0.7)

L_ce1 = CE(Teacher.b[24], Student.b[24])  ← 已有共享层, 冻结
L_ce2 = CE(Teacher.b[26], Student.b[25])  ← 末端对齐
L_ce3 = CE(Teacher.b[22], Student.b[22])  ← 新增共享层, 可训练
```

**可训练参数** (block[22] 新增, ~12K):
```
block[22] 的:
  attn_transform_F1_weight, attn_transform_F2_weight
  mlp_dwconv (weight + bias)
  mlp_transform_norm (weight + bias)
  norm1 (weight + bias)
  norm2 (weight + bias)

block[24] 的已有参数: ★ 全部冻结
```

**训练配置**:
| 参数 | 值 |
|---|---|
| 优化器 | Adam (lr=1e-3) |
| Batch size | 4 |
| Epochs | 10 (steps/epoch=50) |
| Teacher | 原始 RiverOne-QC-4B-v2 (未量化, 全精度 ViT) |
| Student | MiniViT 双重共享模型 |

### 3.4 压缩效果

| 指标 | 压缩前 (27层 ViT) | 压缩后 (MiniViT ×2) |
|---|---|---|
| block 23/24 MSA+MLP | ~14M params | ~12K params |
| block 21/22 MSA+MLP | ~14M params | ~12K params |
| 总计节省 | — | **~28M params → ~24K params** (~1167×) |

---

## Stage 4: VQC 蒸馏 — 量子电路权重补偿 (必选)

### 4.1 动机

MiniViT 权重复用后，block 22/24 的 MLP 权重（fc1, fc2）与源 block 21/23 共享，虽然变换矩阵提供了一定补偿，但 fc1/fc2 的实际权重值与原始 Teacher block 仍有偏差。**Stage 4 是必选步骤**，使用 VQC（Variational Quantum Circuit）+ HyperNetwork 为共享块生成合成 MLP 权重，恢复精度至接近 Teacher 水平。

### 4.2 三阶段流程

```
Phase 1a (train_vqc_fc1.py)           Phase 1b (train_vqc_fc2.py)
  VQC+Hyper 重建 fc1 权重                VQC+Hyper 重建 fc2 权重
  损失: L_act + λ·L_weight              损失: L_act + λ·L_weight
  ↓                                     ↓
  vqc_we8q6l_b{blk}_state.pt           vqc_we8q6l_b{blk}_fc2_state.pt

                    ↓
Phase 2 (distill_vqc_dual.py)
  全前向蒸馏: MSE(ViT_output_student, ViT_output_teacher)
  仅训练 VQC angles + Hyper (encoder+student 冻结)
  ↓
  vqc_we8q6l_b{blk}_distilled_v3_state.pt

                    ↓
Phase 3 (inject_vqc.py)
  注入 VQC 生成权重到模型 safetensors
  生成修改后的 modeling_ising_vit.py
  ↓
  miniViT_21_24_distilled_vqc/
```

### 4.2 架构 (8q/6l Dual VQC)

```
Teacher fc1/fc2 权重
        │
        ▼
┌──────────────────────┐
│ WeightEncoder         │  统计特征提取 (mean, std, L1)
│ → VQC input (256-d)  │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ TQVQC (8 qubits)     │  Amplitude Encoding → RX/RY/RZ (6 layers)
│ · Ring entanglement  │  → CNOT ring → Pauli X/Y/Z 测量
│ · Feature re-upload  │  → 24-dim quantum feature
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ BlockHyper (×8)      │  低秩分解生成权重块
│ · U generator        │  W_block = U @ V^T / √rank
│ · V generator        │  rank=32
│ · Positional embed   │
└──────────────────────┘
        │
        ▼
  VQC-generated fc1/fc2 weight (replaces shared weight)
```

**关键参数**:
| 参数 | 值 |
|---|---|
| Qubits (n_wires) | 8 |
| Variational layers | 6 |
| Parallel blocks | 8 (split by output rows) |
| Low-rank | 32 |
| Pauli measurement | X, Y, Z on all wires (24-dim) |
| Positional embedding dim | 48 (fc1) / 24 (fc2) |
| 目标 block | 22, 24 |
| 权重类型 | fc1, fc2 (dual) |

### 4.3 训练流程

**Phase 1 — 权重重建** (per-block, per-weight-type):
- 目标: VQC+Hyper 直接重建 Teacher 的 fc1/fc2 权重
- 损失: MSE(VQC_weight, teacher_weight)
- 优化器: AdamW, cosine LR 3e-3→1e-4, 2000 steps
- 输出: `vqc_we8q6l_b{blk}_state.pt`, `vqc_we8q6l_b{blk}_weight.pt`

**Phase 2 — 全前向蒸馏**:
- 目标: 完整 ViT forward 中 MSE(student_ViT_output, teacher_ViT_output)
- 可训练: VQC angles + Hyper (encoder 冻结, student 模型冻结)
- 损失: L_distill + 0.1 * L_recon
- 优化器: AdamW (lr=1e-4, wd=1e-5), warmup + cosine, 500 steps
- 训练数据: 500 张噪声图像 (448×448)
- 输出: `vqc_we8q6l_b{blk}_distilled_v3_state.pt`, `*_weight.pt`

### 4.4 注入

训练完成后，VQC 生成的合成权重作为额外张量注入模型:

```
vision_model.blocks.22.vqc_fc1_weight  [4304, 1152]
vision_model.blocks.22.vqc_fc2_weight  [4304, 4304]
vision_model.blocks.24.vqc_fc1_weight  [4304, 1152]
vision_model.blocks.24.vqc_fc2_weight  [4304, 4304]
```

推理时，block 22/24 的 MLP 前向使用这些 VQC 权重替代共享的 block 21/23 MLP 权重。

### 4.5 使用方法

```bash
cd paramgen/

# Phase 1a: fc1 权重重建 (per block)
python train_vqc_fc1.py --teacher_dir ../weights/<aqlm_model>

# Phase 1b: fc2 权重重建 (per block)
python train_vqc_fc2.py --teacher_dir ../weights/<aqlm_model>

# Phase 2: 全前向双 VQC 蒸馏
python distill_vqc_dual.py \
  --teacher_dir ../weights/<aqlm_model> \
  --student_dir ../weights/miniViT_21_24_distilled

# Phase 3: 注入 VQC 权重到模型
python inject_vqc.py \
  --src_dir ../weights/miniViT_21_24_distilled \
  --out_dir ../weights/miniViT_21_24_distilled_vqc
```

> 📖 模型: `vqc_models.py` | P1a: `train_vqc_fc1.py` | P1b: `train_vqc_fc2.py` | P2: `distill_vqc_dual.py` | P3: `inject_vqc.py`

---

## 完整执行流程

### 前置条件

```bash
# 环境
pip install torch>=2.1.0 transformers>=4.38.0 aqlm>=1.1.0 safetensors tqdm pillow torchvision

```
RiverOne-QC-4B-v1-AQLM-miniViT/
├── AQLM/                   # 已量化模型 (2×16, L8-L32, 175 AQLM 层)
├── miniViT/                # MiniViT 压缩工作区
├── PV-tuning/              # PV-Tuning 恢复
└── vqc/                    # VQC 参数生成 (可选)
```

### Step 1: AQLM 量化 (2×16, L8-L32)

```bash
cd quantize/
# 可通过环境变量覆盖路径:
#   RIVERONE_SOURCE_MODEL=/path/to/source/model
#   RIVERONE_OUTPUT_DIR=/path/to/output
#   RIVERONE_CALIBRATION_JSONL=/path/to/qcaleval_zs_sft.jsonl
#   RIVERONE_IMAGE_BASE=/path/to/images
python quantize.py  # ~2-3.5h on A100

# 输出: weights/RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP/
```

### Step 2: PV-Tuning 精度恢复

```bash
cd finetune/

export CUDA_VISIBLE_DEVICES=7,6,5,4
export MODEL_DIR=../weights/RiverOne-QC-4B-MPO-AQLM-2x16-L8L32-AttnMLP

python3 train_pv_tuning.py \
    --model_dir "${MODEL_DIR}" \
    --data_dir ./QcalEval \
    --train_files qcaleval_zs_sft.jsonl \
    --output_dir outputs/pv_fc_attnmlp_zs_sft \
    --num_gpus 4 \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --lr 3e-4 \
    --code_lr 1e-2 \
    --max_code_change_per_step 1e-3 \
    --beam_size 1 \
    --lr_scheduler cosine \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --log_every_steps 1 \
    --save_every_steps 200

# 输出: outputs/pv_fc_attnmlp_zs_sft/
```

### Step 3: MiniViT Stage 1 — 第一对权重复用 (23→24)

```bash
cd compress/
# 环境变量: MINIVIT_SOURCE_MODEL (默认指向 quantize 输出)
python apply_minivit_23_24.py       # 权重复用
# 环境变量: MINIVIT_TEACHER_MODEL (默认指向 quantize 输出)
python distill_minivit_23_24.py     # 蒸馏 (--epochs 10 --batch-size 4 --lr 1e-3)
python verify_minivit.py --check minivit

# 输出: weights/miniViT_23_24_distilled/
```

### Step 4: MiniViT Stage 2 — 第二对权重复用 (21→22, 叠加)

```bash
cd compress/
# 环境变量: MINIVIT_STAGE1_DISTILLED (默认指向 stage 1 蒸馏输出)
python apply_minivit_21_22.py       # 基于蒸馏后模型叠加第二对
# 环境变量: MINIVIT_TEACHER_MODEL, MINIVIT_DATA_DIR
python distill_minivit_21_22.py     # 5项蒸馏 (--epochs 10 --steps-per-epoch 50 --lr 1e-3)
python verify_minivit.py --check distilled

# 输出: weights/miniViT_21_24_distilled/
```

### Step 5: VQC 蒸馏补偿 (必选)

```bash
cd paramgen/

# Phase 1a: fc1 权重重建
python train_vqc_fc1.py --teacher_dir ../weights/<aqlm_model>

# Phase 1b: fc2 权重重建
python train_vqc_fc2.py --teacher_dir ../weights/<aqlm_model>

# Phase 2: 双 VQC 蒸馏
python distill_vqc_dual.py \
  --teacher_dir ../weights/<aqlm_model> \
  --student_dir ../weights/miniViT_21_24_distilled

# Phase 3: 注入 VQC 权重
python inject_vqc.py \
  --src_dir ../weights/miniViT_21_24_distilled \
  --out_dir ../weights/miniViT_21_24_distilled_vqc

# 输出: weights/miniViT_21_24_distilled_vqc/
```

### Step 6: 评估

```bash
python3 evaluate_perplexity.py \
    --model_dir outputs/pv_fc_attnmlp_zs_sft_miniViT_21_24_distilled \
    --output_json eval_results/after_pv.json
```

---

## 模型架构全图

```
pv_fc_attnmlp_zs_sft_miniViT_21_24_distilled
│
├── vision_model: IsingVisionEncoder (27 blocks)
│   ├── blocks[0..20]: 正常 block
│   ├── blocks[21]: 源块 ←─┐ 共享 (MiniViT pair 2)
│   ├── blocks[22]: 目标块 ←┘ 变换矩阵 F1,F2,dwconv,transform_norm
│   ├── blocks[23]: 源块 ←─┐ 共享 (MiniViT pair 1)
│   ├── blocks[24]: 目标块 ←┘ 变换矩阵 F1,F2,dwconv,transform_norm (已蒸馏冻结)
│   ├── blocks[25..26]: 正常 block
│   └── merger: IsingPatchMerger
│
├── mlp1: Linear(2048→2560) + GELU + Linear(2560→2560)
│
└── language_model: Qwen3ForCausalLM (36 layers)
    ├── embed_tokens: [151936, 2560]
    ├── layers.0..7                      ← bf16 保留 (早期层)
    ├── layers.8..32  × 7 个 AQLM 2×16 线性投影 (PV-Tuned)  ← 量化
    ├── layers.33..35                    ← bf16 保留 (晚期层)
    ├── norm: RMSNorm
    └── lm_head: [2560, 151936]
```

---

## 参数统计

> 来源: `param_count.json` (pv_fc_attnmlp_zs_sft_miniViT)

| 组件 | 参数量 | 格式 |
|---|---|---|
| LLM embed_tokens | 388.96M | bf16 |
| LLM lm_head | 388.96M | bf16 |
| LLM 未量化层 L0-7, L33-35 | 370.29M | bf16 |
| LLM AQLM codebooks L8-32 (×175) | 367.00M | int32 |
| LLM AQLM scales L8-32 (×175) | 0.77M | fp32 |
| LLM LayerNorms | 0.18M | bf16 |
| LLM final_norm | 2.56K | bf16 |
| ViT 25 effective blocks | 380.99M | bf16 |
| ViT patch_embed + pos | 14.30M | bf16 |
| ViT merger | 20.82M | bf16 |
| ViT MiniViT transforms (×2) | 7.2K | bf16 |
| MLP1 projector | 17.05M | bf16 |
| **总计** | **3.00B** | — |

> AQLM 2×16 方案: 每层 7 个子层 × 2 codebooks × 65536 entries × 16 dims → codebook 参数较 1×16 翻倍
> MiniViT ×2 对: block 23/24 + 21/22 节省 ~28M params，新增 ~7.2K transform params

---

## 参考文献

1. **PV-Tuning**: Malinovskii et al. "PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression." arXiv:2405.14852, 2024.
2. **AQLM**: Egiazarian et al. "Extreme Compression of Large Language Models via Additive Quantization." ICML 2024.
3. **MiniViT**: Zhang et al. "MiniViT: Compressing Vision Transformers with Weight Multiplexing." CVPR 2022.
4. **Qwen3**: Qwen Team. "Qwen3 Technical Report." 2025.
5. **InternVL 3.5**: Chen et al. "InternVL 3.5." 2025.
