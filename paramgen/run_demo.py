#!/usr/bin/env python3
"""
VQC-MLPNet Demo — VQC-based Neural Network Weight Parameter Generation
=======================================================================
Demonstrates:
  1. Pure weight generation via VQC
  2. End-to-end forward propagation
  3. Training loop with classification loss
  4. Inference and weight analysis

Usage:
  cd paramgen
  pip install -r requirements.txt
  python run_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

import numpy as np
import time

from models import VQC_MLPNet
from utils import count_parameters, print_weight_stats

# ============== Random seed ==============
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# ============== Device detection ==============
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[INFO] Running on CPU")


if __name__ == "__main__":
    print("=" * 60)
    print("  VQC-MLPNet — Neural Network Parameter Generation Demo")
    print("=" * 60)

    # ---- Hyperparameters ----
    N_WIRES = 12               # Number of qubits
    N_QLAYERS = 3              # VQC variational layers
    WEIGHT_SHAPE = (4304, 1152)  # Target MLP weight shape
    RANK = 64                  # Low-rank decomposition rank
    LATENT_DIM = 256           # HyperNetwork hidden dim
    OUT_FEATURES = 2           # Final output classes
    FEATURE_DIM = 2 ** N_WIRES  # AmplitudeEncoder input dim = 2^n_wires
    BATCH_SIZE = 4             # Batch size
    NUM_EPOCHS = 200           # Training epochs

    print(f"\n[Config]")
    print(f"  Weight shape:  {WEIGHT_SHAPE}  ({WEIGHT_SHAPE[0] * WEIGHT_SHAPE[1]:,} params)")
    print(f"  Low-rank r:    {RANK}")
    print(f"  Qubits:        {N_WIRES}")
    print(f"  VQC layers:    {N_QLAYERS}")
    print(f"  Feature dim:   {FEATURE_DIM}  (= 2^{N_WIRES})")
    print(f"  Batch size:    {BATCH_SIZE}")

    # ---- Instantiate model ----
    model = VQC_MLPNet(
        n_wires=N_WIRES,
        n_qlayers=N_QLAYERS,
        weight_shape=WEIGHT_SHAPE,
        rank=RANK,
        latent_dim=LATENT_DIM,
        out_features=OUT_FEATURES,
        noise_prob=0.01,  # 1% depolarizing noise simulating real hardware
    ).to(device)

    print(f"\n[Model] Trainable parameters: {count_parameters(model):,}")

    # ---- Generate synthetic data ----
    # Random feature input (fed to VQC)
    random_features = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
    # Actual data input (transformed using generated W)
    data_input = torch.randn(BATCH_SIZE, WEIGHT_SHAPE[1], device=device)  # [B, 1152]
    data_labels = torch.randint(0, OUT_FEATURES, (BATCH_SIZE,), device=device)

    # ---- Quantum device ----
    q_dev = tq.QuantumDevice(n_wires=N_WIRES, bsz=BATCH_SIZE, device=device)

    # ========================================================================
    #  Test 1: Pure weight generation
    # ========================================================================
    print(f"\n{'='*60}")
    print("  Test 1: Pure Weight Generation")
    print(f"{'='*60}")
    with torch.no_grad():
        W_generated = model.generate_weights_only(random_features, q_dev)
    print_weight_stats(W_generated, f"W [{WEIGHT_SHAPE[0]}, {WEIGHT_SHAPE[1]}]")

    # ========================================================================
    #  Test 2: End-to-end forward pass
    # ========================================================================
    print(f"\n{'='*60}")
    print("  Test 2: End-to-End Forward Pass")
    print(f"{'='*60}")
    q_dev.reset_states(BATCH_SIZE)
    output = model(data_input, random_features, q_dev)
    print(f"  Input shape:   {list(data_input.shape)}")
    print(f"  Output shape:  {list(output.shape)}")
    print(f"  Output values: {output}")

    # ========================================================================
    #  Test 3: Training loop
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"  Test 3: Training Loop ({NUM_EPOCHS} epochs)")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Fixed batch of random features for training (simulating "random features as seed")
    train_features = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
    train_data = torch.randn(BATCH_SIZE, WEIGHT_SHAPE[1], device=device)
    train_labels = torch.randint(0, OUT_FEATURES, (BATCH_SIZE,), device=device)

    model.train()
    t_start = time.time()

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        q_dev.reset_states(BATCH_SIZE)

        output = model(train_data, train_features, q_dev)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 40 == 0 or epoch == 0:
            pred = output.argmax(dim=1)
            acc = (pred == train_labels).float().mean().item()
            print(f"  Epoch {epoch+1:4d}/{NUM_EPOCHS}  "
                  f"loss={loss.item():.6f}  acc={acc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    t_end = time.time()
    print(f"\n  Training complete, time: {t_end - t_start:.2f} s")

    # ========================================================================
    #  Test 4: Inference & weight consistency
    # ========================================================================
    print(f"\n{'='*60}")
    print("  Test 4: Inference & Weight Analysis")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        q_dev.reset_states(BATCH_SIZE)
        W_final = model.generate_weights_only(train_features, q_dev)
        output_final = model(train_data, train_features, q_dev)
        pred_final = output_final.argmax(dim=1)
        acc_final = (pred_final == train_labels).float().mean().item()

    print_weight_stats(W_final, f"Final W [{WEIGHT_SHAPE[0]}, {WEIGHT_SHAPE[1]}]")
    print(f"  Final accuracy: {acc_final:.4f}")

    # ---- Save model ----
    save_path = "./vqc_mlpnet_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n  Model saved to: {save_path}")

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")
