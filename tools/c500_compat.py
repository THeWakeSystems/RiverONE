"""
C500 兼容性检测与配置工具

提供统一的设备检测和 C500 适配配置，所有 RiverONE 模块通过此模块
获取设备信息，避免硬编码 "cuda:0"。

MetaX flash_attn 说明:
  本机安装了 MetaX 定制版 flash_attn (2.6.3+metax3.7.1.3torch2.8)。
  - flash_attn_func / flash_attn_varlen_func: ✅ 可用 (forward + backward)
  - FlashAttention / FlashSelfAttention nn.Module: ❌ 未包含在 MetaX 构建中
  - HuggingFace attn_implementation="flash_attention_2": ❌ 不可用 (需要 FlashAttention module)
  - 性能: 中等序列长度 (≤4K) 与 SDPA 接近，长序列 (≥8K) 有 O(1) 显存优势
  - 建议: 默认使用 sdpa；需要长序列或直接调用 flash_attn_func 的场景可手动切换
"""
from __future__ import annotations

import os
import torch
from typing import List, Optional


# ── 缓存的检测结果 ──────────────────────────────────────────
_flash_attn_info: Optional[dict] = None


def get_device_name() -> str:
    """获取当前 GPU 名称（MetaX C500 / NVIDIA A100 / ...）"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def is_metax() -> bool:
    """检测是否为沐曦 MetaX C500 平台"""
    name = get_device_name()
    return "MetaX" in name or "C500" in name or "MACA" in name


def get_default_device() -> torch.device:
    """获取默认计算设备。单 GPU 返回 cuda:0，无 GPU 返回 cpu。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def get_default_devices() -> List[str]:
    """获取所有可用 GPU 设备列表"""
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def configure_tf32(enabled: bool = False) -> None:
    """安全配置 TF32。C500 不支持 TF32，自动跳过。"""
    if not is_metax():
        try:
            torch.backends.cudnn.allow_tf32 = enabled
            torch.backends.cuda.matmul.allow_tf32 = enabled
        except Exception:
            pass


# ── flash_attn 检测 ──────────────────────────────────────────

def detect_flash_attn() -> dict:
    """检测 MetaX flash_attn 可用性，结果缓存。

    Returns:
        {
            "available": bool,          # flash_attn 包是否可导入
            "version": str|None,         # flash_attn 版本号
            "is_metax_build": bool,      # 是否为 MetaX 定制构建
            "flash_attn_func": bool,     # 核心函数可用
            "flash_attn_varlen_func": bool,  # 变长序列函数可用
            "FlashAttention_module": bool,   # nn.Module 是否可用 (需此才能用 attn_implementation)
            "can_use_attn_impl": bool,   # 能否通过 transformers attn_implementation 使用
            "recommendation": str,       # 推荐用法
        }
    """
    global _flash_attn_info
    if _flash_attn_info is not None:
        return _flash_attn_info

    info = {
        "available": False,
        "version": None,
        "is_metax_build": False,
        "flash_attn_func": False,
        "flash_attn_varlen_func": False,
        "FlashAttention_module": False,
        "can_use_attn_impl": False,
        "recommendation": "sdpa",
    }

    try:
        import pkg_resources
        ver = pkg_resources.get_distribution("flash-attn").version
        info["available"] = True
        info["version"] = ver
        info["is_metax_build"] = "metax" in ver.lower()
    except Exception:
        _flash_attn_info = info
        return info

    try:
        from flash_attn import flash_attn_func
        info["flash_attn_func"] = True
    except ImportError:
        pass

    try:
        from flash_attn import flash_attn_varlen_func
        info["flash_attn_varlen_func"] = True
    except ImportError:
        pass

    try:
        from flash_attn.modules.mha import FlashAttention, FlashSelfAttention
        info["FlashAttention_module"] = True
    except ImportError:
        pass

    # 只有 FlashAttention module 可用时才能通过 attn_implementation 使用
    info["can_use_attn_impl"] = info["FlashAttention_module"]

    # 推荐策略
    if info["can_use_attn_impl"]:
        info["recommendation"] = "flash_attention_2"
    elif info["flash_attn_func"] and info["is_metax_build"]:
        info["recommendation"] = (
            "sdpa (flash_attn_func 可直接调用，但 FlashAttention module 不可用，"
            "无法通过 transformers attn_implementation 集成)"
        )
    elif info["flash_attn_func"]:
        info["recommendation"] = "flash_attention_2"
    else:
        info["recommendation"] = "sdpa"

    _flash_attn_info = info
    return info


def get_attn_implementation() -> str | None:
    """获取合适的 attention 实现。

    优先级:
      1. 环境变量 RIVERONE_ATTN_IMPL (手动覆盖)
      2. MetaX 平台: 检测 flash_attn 可用性
         - FlashAttention module 可用 → "flash_attention_2"
         - 仅 flash_attn_func 可用 → "sdpa" (无法通过 HF 集成)
         - 不可用 → "sdpa"
      3. NVIDIA 平台: None (使用 transformers 默认值)
    """
    env_val = os.environ.get("RIVERONE_ATTN_IMPL", None)
    if env_val is not None:
        return env_val

    if is_metax():
        fa_info = detect_flash_attn()
        if fa_info["can_use_attn_impl"]:
            return "flash_attention_2"
        else:
            return "sdpa"

    return None  # NVIDIA 上使用 transformers 默认值


# ── 环境信息打印 ──────────────────────────────────────────────

def print_env_info() -> None:
    """打印当前环境信息（用于调试和日志）"""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Platform: {'MetaX C500' if is_metax() else 'NVIDIA GPU'}")
        print(f"Default device: {get_default_device()}")

        # flash_attn 检测
        fa = detect_flash_attn()
        print(f"\nflash_attn:")
        print(f"  Installed: {fa['available']}")
        if fa['available']:
            print(f"  Version: {fa['version']}")
            print(f"  MetaX build: {fa['is_metax_build']}")
            print(f"  flash_attn_func: {fa['flash_attn_func']}")
            print(f"  FlashAttention module: {fa['FlashAttention_module']}")
            print(f"  HF attn_impl usable: {fa['can_use_attn_impl']}")
            print(f"  Recommendation: {fa['recommendation']}")

        print(f"\nAttention config:")
        print(f"  RIVERONE_ATTN_IMPL env: {os.environ.get('RIVERONE_ATTN_IMPL', 'not set')}")
        print(f"  Auto-detected: {get_attn_implementation()}")
    else:
        print(f"Platform: CPU-only")


if __name__ == "__main__":
    print_env_info()
