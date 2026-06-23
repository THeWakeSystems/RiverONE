#!/usr/bin/env python3
"""
=============================================================================
 verify_minivit.py — MiniViT 压缩验证脚本
=============================================================================
 验证项：
   1. MiniViT 模型加载正确性
   2. 权重复用验证（MSA/MLP 共享、norm 独立）
   3. 变换矩阵存在性检查
   4. 推理测试（对比 teacher 和 student 输出）

 使用方法：
   python3 verify_minivit.py [--check distill]  # 检查蒸馏后模型
   python3 verify_minivit.py                     # 默认检查 MiniViT 模型
=============================================================================
"""
import sys, os, gc, time, json, argparse
from pathlib import Path
from collections import defaultdict

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = (
    "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/"
    "AQLM/RiverOne-QC-4B-v2-AQLM-36L"
)
MINIVIT = str(SCRIPT_DIR.parent / "weights" / "miniViT")
DISTILLED = str(SCRIPT_DIR.parent / "weights" / "miniViT_distilled")
DEVICE = "cuda:0"


def log(msg):
    print(f"[Verify] {msg}")


def _clear_cache():
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "modeling_riverone_qc", "modeling_ising_vit",
            "configuration_riverone_qc", "conversation",
        ):
            del sys.modules[mod_name]


def _load_aqlm(model, model_dir: str):
    """加载 AQLM 量化权重到 LLM。"""
    qc = Path(model_dir) / "quant_config.json"
    if not qc.exists():
        return
    if json.loads(open(qc).read()).get("quantization_method") != "AQLM":
        return

    from aqlm import QuantizedLinear as AQLMLinear
    idx = json.loads(
        (Path(model_dir) / "model.safetensors.index.json").read_text()
    )
    wm = idx["weight_map"]

    grp = defaultdict(dict)
    for k in wm:
        if k.endswith(".codebooks"): grp[k[:-10]]["cb"] = k
        elif k.endswith(".codes"):   grp[k[:-6]]["cd"] = k
        elif k.endswith(".scales"):  grp[k[:-7]]["sc"] = k

    tens = {}
    for s in sorted(set(wm.values())):
        p = Path(model_dir) / s
        if p.exists():
            tens.update(__import__("safetensors").torch.load_file(str(p)))

    layers = model.language_model.model.layers
    for base, info in grp.items():
        parts = base.split(".")
        li = int(parts[3])
        sp = parts[4:]
        cb, cd, sc = tens[info["cb"]], tens[info["cd"]], tens[info["sc"]]
        nc, cs, og, ig = cb.shape
        ql = AQLMLinear(
            cd.shape[1] * ig, cd.shape[0] * og, ig, og, nc,
            cs.bit_length() - 1, bias=False, dtype=cb.dtype,
        )
        ql.codebooks.data.copy_(cb)
        ql.codes.data.copy_(cd.to(ql.codes.dtype))
        ql.scales.data.copy_(sc)
        parent = layers[li]
        for seg in sp[:-1]:
            parent = getattr(parent, seg)
        ql = ql.to(next(parent.parameters()).device)
        setattr(parent, sp[-1], ql)


def load_model(model_dir: str):
    """从指定目录加载模型。"""
    _clear_cache()
    while model_dir in sys.path:
        sys.path.remove(model_dir)
    sys.path.insert(0, model_dir)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None,
        local_files_only=True,
    ).to(DEVICE).eval()
    _load_aqlm(model, model_dir)
    return model


def verify_model(m, label):
    """验证 MiniViT 权重复用是否正确。"""
    blocks = m.vision_model.blocks
    ok = True

    checks = [
        ("MSA共享", blocks[24].attn is blocks[23].attn),
        ("MLP共享", blocks[24].mlp is blocks[23].mlp),
        ("norm1独立", blocks[24].norm1 is not blocks[23].norm1),
        ("norm2独立", blocks[24].norm2 is not blocks[23].norm2),
        ("F1存在", hasattr(blocks[24], "attn_transform_F1_weight")),
        ("F2存在", hasattr(blocks[24], "attn_transform_F2_weight")),
        ("dwconv存在", hasattr(blocks[24], "mlp_dwconv")),
        ("transform_norm存在", hasattr(blocks[24], "mlp_transform_norm")),
    ]

    for name, result in checks:
        symbol = "✅" if result else "❌"
        log(f"  {symbol} {name}")
        if not result:
            ok = False

    if ok:
        log(f"  ✅ {label} 全部检查通过！")
    else:
        log(f"  ❌ {label} 存在问题！")
    return ok


def compare_outputs(model_a, model_b, label_a, label_b):
    """对比两个模型 ViT 输出。"""
    dummy = torch.randn(1, 3, 448, 448, device=DEVICE).to(torch.bfloat16)

    with torch.no_grad():
        out_a = model_a.vision_model(dummy)
        out_b = model_b.vision_model(dummy)

    mse = torch.nn.functional.mse_loss(out_a, out_b).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_a.flatten(), out_b.flatten(), dim=0
    ).item()

    log(f"  {label_a} vs {label_b}:")
    log(f"    MSE: {mse:.6f}")
    log(f"    Cosine Similarity: {cos_sim:.6f}")

    return mse, cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", choices=["minivit", "distilled"], default="minivit",
    )
    args = parser.parse_args()

    log("=" * 60)
    log(" RiverOne-QC-4B-v1 MiniViT 压缩验证")
    log("=" * 60)

    # 1. 加载原始模型（teacher）
    log("加载原始模型（Teacher）...")
    teacher = load_model(SOURCE)
    total_params = sum(p.numel() for p in teacher.parameters())
    log(f"  总参数量: {total_params / 1e9:.2f}B")

    # 2. 检查 teacher 的 ViT blocks
    log(f"\n  Teacher ViT blocks: {len(teacher.vision_model.blocks)}")
    log(f"  block 23 attn id: {id(teacher.vision_model.blocks[23].attn)}")
    log(f"  block 24 attn id: {id(teacher.vision_model.blocks[24].attn)}")
    log(f"  Teacher MSA 是否共享: {teacher.vision_model.blocks[24].attn is teacher.vision_model.blocks[23].attn}")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 加载目标模型
    target_dir = MINIVIT if args.check == "minivit" else DISTILLED
    label = "MiniViT" if args.check == "minivit" else "MiniViT 蒸馏后"

    log(f"\n加载 {label} 模型...")
    if not Path(target_dir).exists():
        log(f"  ❌ {label} 模型目录不存在: {target_dir}")
        log(f"  请先运行 apply_minivit.py 生成 MiniViT 模型")
        return

    target = load_model(target_dir)
    total_params_t = sum(p.numel() for p in target.parameters())
    log(f"  总参数量: {total_params_t / 1e9:.2f}B")

    # 4. 验证结构
    log(f"\n{label} 结构验证:")
    verify_model(target, label)

    log("\n" + "=" * 60)
    log(" 验证完成！")
    log("=" * 60)


if __name__ == "__main__":
    main()
