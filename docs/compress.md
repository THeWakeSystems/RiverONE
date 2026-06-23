# RiverOne-QC-4B-v1 MiniViT 视觉编码器压缩项目

## 概述

本项目对 **RiverOne-QC-4B-v1**（已完成 AQLM 量化的版本）的视觉编码器（Ising ViT）
应用 **MiniViT** 权重复用压缩（CVPR 2022），进一步减少模型参数量。

### MiniViT 原理

MiniViT 发现相邻 Transformer 层的 MSA/MLP 权重高度相似，通过以下方式压缩：

1. **权重复用**：让相邻两层共享 MSA/MLP 权重
2. **轻量变换矩阵**：用 ~12K 参数补偿共享带来的精度损失
   - F1, F2：注意力变换矩阵（16×16 各 256 参数）
   - Depthwise Conv：MLP 输入变换（~4.6K 参数）
   - LayerNorm + Transform Norm
3. **蒸馏训练**：用原始 ViT 作为 teacher 蒸馏 student 的变换矩阵

### 压缩范围

| 项目 | 详情 |
|------|------|
| 源模型 | RiverOne-QC-4B-v1-AQLM-36L（LLM 全部36层已 AQLM 量化） |
| 视觉编码器 | Ising ViT × 27 blocks |
| 压缩目标 | block 23 → block 24 权重复用（倒数第3、4层） |
| 共享组件 | MSA (qkv + proj) + MLP (fc1 + fc2) |
| 独立组件 | norm1, norm2（LayerNorm 保持独立） |
| 新增参数 | ~12K（F1, F2, dwconv, transform_norm） |
| 节省参数 | ~14M（一对 MSA + MLP 权重） |

---

## 目录结构

```
RiverOne-QC-4B-v1-AQLM-miniViT/
├── scripts/                          ← AQLM 量化脚本（已有）
├── aqlm_lib/                         ← AQLM 引擎（已有）
├── RiverOne-QC-4B-v1-AQLM-36L/      ← 量化后模型（已有）
└── miniViT/                          ← ★ MiniViT 压缩工作区
    ├── scripts/                      ← MiniViT 脚本
    │   ├── apply_minivit.py          ★ 权重复用 + 模型保存
    │   ├── distill_minivit.py        ★ 蒸馏训练
    │   ├── verify_minivit.py         ★ 验证脚本
    │   └── README.md                 ← 本文件
    ├── miniViT/                      ← 权重复用后模型（apply 输出）
    └── miniViT_distilled/            ← 蒸馏后模型（distill 输出）
```

---

## 使用流程

### Step 1: 应用权重复用

```bash
cd /home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/miniViT/scripts
python3 apply_minivit.py
```

**说明**：加载 AQLM 量化模型 → block 23→24 权重复用 → 生成修改后的 modeling_ising_vit.py → 保存到 `../miniViT/`

### Step 2: 验证权重复用

```bash
python3 verify_minivit.py
```

**检查项**：MSA/MLP 共享、norm 独立、变换矩阵存在、模型可正常推理

### Step 3: 蒸馏训练

```bash
python3 distill_minivit.py --epochs 10 --batch-size 4 --lr 1e-3
```

**说明**：用原始 ViT（teacher）蒸馏 MiniViT（student）的变换矩阵，优化 ~12K 参数

### Step 4: 验证蒸馏效果

```bash
python3 verify_minivit.py --check distilled
```

---

## 注意事项

1. **LLM 不受影响**：MiniViT 仅修改视觉编码器的 block 23/24，LLM 的 AQLM 量化完全保留
2. **显存需求**：蒸馏需同时加载 teacher 和 student，建议 ≥24GB 显存
3. **训练数据**：当前使用随机图像蒸馏（无标注），可替换为真实图像
4. **参考论文**：MiniViT: Compressing Vision Transformers with Weight Multiplexing (CVPR 2022)

---

## 相关资源

- MiniViT 论文: https://arxiv.org/abs/2204.07154
- 参考实现: `/home/hyba/lyc/RiverOne-QC-4B-AQLM-36L-last34-miniViT/`
- AQLM 量化: `/home/hyba/lyc/RiverOne-QC-4B-v1-AQLM-miniViT/scripts/`
