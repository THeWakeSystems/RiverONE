"""
=============================================================================
VQC Weight Encoder — Shared Model Definitions (Stage 4)
=============================================================================
Production VQC architecture for MiniViT shared-block MLP weight compensation.

Architecture (8q/6l):
  WeightEncoder → VQC (8 qubits, 6 layers, ring entanglement) → BlockHyper
  → Low-rank weight reconstruction (rank=32, 8 parallel blocks)

Components:
  TQVQC             — Variational quantum circuit (amplitude encoding + RX/RY/RZ + CNOT)
  BlockHyper        — Low-rank weight matrix generator (U @ V^T / √rank)
  WeightEncoder     — Statistical feature extractor (mean/std/L1 → VQC input)
  VQCWeightGenerator — Full pipeline: encoder → N×VQCs → N×Hypers → concat

Used by:
  train_vqc_fc1.py      — Phase 1: fc1 weight reconstruction
  train_vqc_fc2.py      — Phase 1: fc2 weight reconstruction
  distill_vqc_dual.py   — Phase 2: full-forward dual VQC distillation
  inject_vqc.py         — Phase 3: weight injection into model
=============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical


# ═══════════════════════════════════════════════════════════════
#  Pauli measurement strings (X, Y, Z on all wires)
# ═══════════════════════════════════════════════════════════════

def _build_pauli_strings(n_wires: int) -> list[str]:
    strings = []
    for basis in ["X", "Y", "Z"]:
        for i in range(n_wires):
            s = ["I"] * n_wires
            s[i] = basis
            strings.append("".join(s))
    return strings


# ═══════════════════════════════════════════════════════════════
#  TQVQC — TorchQuantum Variational Quantum Circuit
# ═══════════════════════════════════════════════════════════════

class TQVQC(tq.QuantumModule):
    """Variational quantum circuit with data re-uploading and XYZ measurement.

    Per layer: RX/RY/RZ on each wire → CNOT ring entanglement.
    Periodic feature re-upload every `reupload_every` layers.
    Measurement: Pauli X, Y, Z expectation values on all wires.
    """

    def __init__(self, n_wires: int, n_qlayers: int, reupload_every: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.reupload_every = reupload_every
        self.encoder = tq.AmplitudeEncoder()
        self.pauli_strings = _build_pauli_strings(n_wires)

        self.variational = nn.ModuleDict({
            f"l{k}_w{i}": nn.ModuleDict({
                "rx": tq.RX(has_params=True, trainable=True),
                "ry": tq.RY(has_params=True, trainable=True),
                "rz": tq.RZ(has_params=True, trainable=True),
            })
            for k in range(n_qlayers) for i in range(n_wires)
        })

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.q_device = q_device
        self.encoder(self.q_device, x)

        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                g = self.variational[f"l{k}_w{i}"]
                g["rx"](self.q_device, wires=i)
                g["ry"](self.q_device, wires=i)
                g["rz"](self.q_device, wires=i)

            for i in range(self.n_wires - 1):
                tqf.cnot(self.q_device, wires=[i, i + 1],
                         static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(self.q_device, wires=[self.n_wires - 1, 0],
                     static=self.static_mode, parent_graph=self.graph)

            if (k + 1) % self.reupload_every == 0 and k < self.n_qlayers - 1:
                self.encoder(self.q_device, x)

        exp_vals = [expval_joint_analytical(self.q_device, s).unsqueeze(-1)
                    for s in self.pauli_strings]
        return torch.cat(exp_vals, dim=-1)


# ═══════════════════════════════════════════════════════════════
#  BlockHyper — Low-rank weight block generator
# ═══════════════════════════════════════════════════════════════

class BlockHyper(nn.Module):
    """HyperNetwork: VQC output → low-rank weight matrix [block_out, weight_in].

    Uses learned positional embeddings for row-wise differentiation.
    W = U @ V^T / √rank
    """

    def __init__(self, in_dim: int, block_out: int, weight_in: int,
                 rank: int, latent: int, pd: int = 48):
        super().__init__()
        self.block_out = block_out
        self.weight_in = weight_in
        self.rank = rank

        self.gen_u = nn.Sequential(
            nn.Linear(in_dim + pd, latent), nn.ReLU(),
            nn.Linear(latent, latent), nn.ReLU(),
            nn.Linear(latent, rank),
        )
        self.gen_v = nn.Sequential(
            nn.Linear(in_dim + pd, latent), nn.ReLU(),
            nn.Linear(latent, latent), nn.ReLU(),
            nn.Linear(latent, rank),
        )
        self.pos_u = nn.Parameter(torch.randn(block_out, pd) * 0.02)
        self.pos_v = nn.Parameter(torch.randn(weight_in, pd) * 0.02)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        b, dv = q.shape[0], q.device

        qe = q.unsqueeze(1).expand(-1, self.block_out, -1)
        pu = self.pos_u.unsqueeze(0).expand(b, -1, -1).to(dv)
        U = self.gen_u(torch.cat([qe, pu], -1).reshape(-1, qe.shape[-1] + pu.shape[-1]))
        U = U.view(b, self.block_out, self.rank)

        qe = q.unsqueeze(1).expand(-1, self.weight_in, -1)
        pv = self.pos_v.unsqueeze(0).expand(b, -1, -1).to(dv)
        V = self.gen_v(torch.cat([qe, pv], -1).reshape(-1, qe.shape[-1] + pv.shape[-1]))
        V = V.view(b, self.weight_in, self.rank)

        return torch.bmm(U, V.transpose(1, 2)).mean(0) / (self.rank ** 0.5)


# ═══════════════════════════════════════════════════════════════
#  WeightEncoder — Statistical feature extraction
# ═══════════════════════════════════════════════════════════════

class WeightEncoder(nn.Module):
    """Encodes weight matrix statistics into VQC input features.

    Extracts per-row: mean, std, L1-norm → projects → fuses → VQC input.
    """

    def __init__(self, out_dim: int, vqc_dim: int,
                 stat_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.proj_mean = nn.Sequential(nn.Linear(out_dim, stat_dim), nn.ReLU())
        self.proj_std = nn.Sequential(nn.Linear(out_dim, stat_dim), nn.ReLU())
        self.proj_l1 = nn.Sequential(nn.Linear(out_dim, stat_dim), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(stat_dim * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, vqc_dim),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        m = self.proj_mean(w.mean(dim=1))
        s = self.proj_std(w.std(dim=1, unbiased=False))
        l1 = self.proj_l1(w.abs().mean(dim=1))
        return self.fusion(torch.cat([m, s, l1], dim=0).unsqueeze(0))


# ═══════════════════════════════════════════════════════════════
#  VQCWeightGenerator — Full VQC weight generation pipeline
# ═══════════════════════════════════════════════════════════════

class VQCWeightGenerator(nn.Module):
    """WeightEncoder → VQCs×N_BLOCKS → Hypers×N_BLOCKS → concat → weight.

    Splits output rows across N_BLOCKS parallel VQC+Hyper units.
    """

    def __init__(self, n_wires: int, n_qlayers: int,
                 weight_shape: tuple[int, int],
                 rank: int, latent: int, pd: int = 48,
                 reupload_every: int = 2, n_blocks: int = 8,
                 stat_dim: int = 128, enc_hidden: int = 512):
        super().__init__()
        out_dim, in_dim = weight_shape
        self.n_wires = n_wires
        self.n_blocks = n_blocks

        vqc_dim = 1 << n_wires           # 256
        hyper_in = n_wires * 3            # 24 (XYZ per wire)

        self.encoder = WeightEncoder(out_dim, vqc_dim, stat_dim, enc_hidden)

        base = out_dim // n_blocks
        remainder = out_dim % n_blocks
        self.block_sizes = [base + (1 if b < remainder else 0) for b in range(n_blocks)]

        self.vqcs = nn.ModuleList([
            TQVQC(n_wires, n_qlayers, reupload_every) for _ in range(n_blocks)
        ])
        self.hypers = nn.ModuleList([
            BlockHyper(hyper_in, self.block_sizes[b], in_dim, rank, latent, pd)
            for b in range(n_blocks)
        ])
        self.q_devices = [None] * n_blocks

    def _ensure_qdev(self, idx: int, bsz: int, dv: torch.device):
        if self.q_devices[idx] is None:
            self.q_devices[idx] = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=bsz, device=dv)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        dv = w.device
        x = self.encoder(w)       # [1, 256]
        bsz = x.shape[0]

        blocks = []
        for b in range(self.n_blocks):
            self._ensure_qdev(b, bsz, dv)
            self.q_devices[b].reset_states(bsz)
            q_out = self.vqcs[b](self.q_devices[b], x)    # [1, 24]
            W_block = self.hypers[b](q_out)                 # [block_size, in_dim]
            blocks.append(W_block)

        return torch.cat(blocks, dim=0)
