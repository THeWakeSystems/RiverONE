# RiverOne-QC-4B-v1 Compression Pipeline

Extreme compression of the RiverOne-QC-4B-v1 multimodal LLM (~4B params) down to
**~1.72B storage elements (~3.2 GB)** through a three-stage pipeline:

```
Original (bf16, ~8.9 GB)
    ↓ AQLM 1×16
Quantized (~4.5 GB, 36 LLM layers at ~1 bit/param)
    ↓ MiniViT distillation
Vision-compressed (~3.2 GB, 26 visual layers with weight sharing)
    ↓ PV-Tuning
Accuracy-recovered (~3.2 GB, codebooks/codes fine-tuned on QcalEval)
```

## Directory Structure

```
RiverOne/
├── engine/            AQLM quantization engine (core library)
├── quantize/          Quantization configuration scripts
├── compress/          MiniViT vision encoder weight multiplexing
├── finetune/          PV-Tuning accuracy recovery
├── tools/             Utility scripts (dequantize, analyze, swap)
├── weights/           Model weight outputs (excluded from repo)
├── docs/              Full documentation
├── logs/              Archived run logs (compressed summary)
├── README.md
└── .gitignore
```

## Quick Start

### 1. AQLM Quantization

Quantize all 36 LLM transformer layers with AQLM 1×16 scheme:

```bash
cd quantize
pip install -r requirements.txt
python quantize.py
```

See [docs/quantize.md](docs/quantize.md) for details.

### 2. MiniViT Compression

Apply weight multiplexing to the vision encoder (block 23→24 sharing):

```bash
cd compress
python apply_minivit.py      # Weight sharing
python distill_minivit.py     # Distillation training
python verify_minivit.py      # Verification
```

Outputs go to `weights/miniViT_v2/` and `weights/miniViT_v2_distilled/`.

See [docs/compress.md](docs/compress.md) for details.

### 3. PV-Tuning Recovery

Fine-tune quantized codebooks/scales/codes on QcalEval SFT data:

```bash
cd finetune
pip install -r requirements.txt
bash run_pv_tuning.sh
```

See [docs/finetune.md](docs/finetune.md) and [docs/PV_TUNING_TECHNICAL_DOC.md](docs/PV_TUNING_TECHNICAL_DOC.md) for full documentation.

## Key Techniques

### AQLM (Additive Quantization of Language Models)
- **Scheme**: 1×16 (1 codebook, out_group_size=1, in_group_size=16)
- **Codebook size**: 65,536 (16-bit codes)
- **Effective bit-width**: ~1 bit/param
- **Scope**: All 36 LLM decoder layers × 7 linear projections = 252 quantized matrices
- **Preserved at BF16**: Vision encoder, mlp1 multimodal projector, embeddings, LM head, norm layers

### MiniViT
- **Principle**: Adjacent ViT layers have highly similar MSA/MLP weights
- **Implementation**: Block 23/24 share weights + lightweight transform matrices (F1, F2, dwconv)
- **Parameter savings**: ~14M for one block pair

### PV-Tuning
- **P step**: Fix code assignments, optimize continuous codebooks/scales via backprop
- **V step**: Fix codebook values, update discrete codes via L2 beam search in top-τ subspace
- **Convergence guarantee** (Theorem 3.1): Loss monotonically non-increasing

## Requirements

- Python 3.10+
- CUDA 12.1+ / NVIDIA GPU (≥24 GB VRAM recommended)
- Linux (tested on Ubuntu 20.04/22.04)

Core dependencies: `torch>=2.1.0`, `transformers>=4.38.0`, `safetensors`, `aqlm>=1.1.0`

## References

- [AQLM: Extreme Compression of LLMs via Additive Quantization](https://arxiv.org/abs/2401.06118)
- [MiniViT: Compressing Vision Transformers with Weight Multiplexing](https://arxiv.org/abs/2204.07154)
- [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852)
- RiverOne-QC-4B-v1: Internal multimodal model based on Qwen3-4B + Ising Vision Encoder
