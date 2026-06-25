"""
VQC Parameter Generation — Utility Functions
=============================================
Helper functions for parameter counting and weight matrix statistics.
"""

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_weight_stats(W: torch.Tensor, name: str = "W"):
    """Print weight matrix statistics."""
    print(f"\n{'='*60}")
    print(f"  Weight matrix {name}  shape: {list(W.shape)}")
    print(f"  mean={W.mean().item():.6f}  std={W.std().item():.6f}")
    print(f"  min={W.min().item():.6f}  max={W.max().item():.6f}")
    print(f"{'='*60}")
