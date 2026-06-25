"""
VQC Parameter Generation — Model Definitions
=============================================
VQCWeightGenerator: Variational Quantum Circuit that generates MLP weight matrices.
VQC_MLPNet: End-to-end network combining VQC weight generation with MLP inference.

Architecture:
  Random features → VQC amplitude encoding + variational layers → quantum measurement
  → HyperNetwork → low-rank factors U, V → weight matrix W = U @ V^T

Low-rank decomposition avoids directly outputting ~5M parameters:
  W ≈ U @ V^T,  U ∈ R^{out_dim × r}, V ∈ R^{in_dim × r}, rank r ≪ min(out_dim, in_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# TorchQuantum
import torchquantum as tq
import torchquantum.functional as tqf


class VQCWeightGenerator(tq.QuantumModule):
    """
    VQC-based MLP weight generator.

    Pipeline:
      1. Random features encoded to quantum state via amplitude encoding
      2. Multi-layer variational quantum circuit (RX, RY, RZ + CNOT ring entanglement)
         with optional depolarizing noise
      3. PauliZ measurement yields quantum feature vector
      4. HyperNetwork expands quantum features into low-rank factors U, V
      5. Combine: weight matrix W = U @ V^T
    """

    def __init__(self,
                 n_wires: int = 12,
                 n_qlayers: int = 3,
                 weight_shape: tuple = (4304, 1152),
                 rank: int = 64,
                 latent_dim: int = 256,
                 noise_prob: float = 0.0):
        """
        Args:
            n_wires:      Number of qubits in the quantum circuit
            n_qlayers:    Number of variational layers
            weight_shape: Target weight matrix shape (out_features, in_features)
            rank:         Low-rank decomposition rank
            latent_dim:   Hidden dimension of the hypernetwork
            noise_prob:   Depolarizing noise probability per layer (0 = no noise)
        """
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.out_dim, self.in_dim = weight_shape
        self.rank = rank
        self.noise_prob = noise_prob

        # ---- Quantum encoder: encode input features into quantum state ----
        # AmplitudeEncoder: input feature dim must be 2^n_wires
        self.encoder = tq.AmplitudeEncoder()

        # ---- Trainable variational parameters ----
        self.params = nn.ModuleDict({
            f"layer_{k}_wire_{i}": nn.ModuleDict({
                "rx": tq.RX(has_params=True, trainable=True),
                "ry": tq.RY(has_params=True, trainable=True),
                "rz": tq.RZ(has_params=True, trainable=True),
            })
            for k in range(n_qlayers) for i in range(n_wires)
        })

        # ---- Quantum measurement ----
        self.measure = tq.MeasureAll(tq.PauliZ)

        # ---- HyperNetwork: expand quantum measurement into weight matrix factors ----
        # Quantum measurement output dim = n_wires
        self.hyper_net = nn.Sequential(
            nn.Linear(n_wires, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        # Generate low-rank factors U and V separately
        self.fc_U = nn.Linear(latent_dim, self.out_dim * rank)   # → U: [out_dim, rank]
        self.fc_V = nn.Linear(latent_dim, self.in_dim * rank)    # → V: [in_dim, rank]

        # Cache the last generated weight (for inference reuse)
        self._cached_W = None

    def apply_variational_layer(self, k: int):
        """Apply rotation gates for the k-th variational layer."""
        for i in range(self.n_wires):
            gates = self.params[f"layer_{k}_wire_{i}"]
            gates["rx"](self.q_device, wires=i)
            gates["ry"](self.q_device, wires=i)
            gates["rz"](self.q_device, wires=i)

    def apply_entanglement(self):
        """CNOT ring entanglement."""
        for i in range(self.n_wires - 1):
            tqf.cnot(self.q_device, wires=[i, i + 1],
                     static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(self.q_device, wires=[self.n_wires - 1, 0],
                 static=self.static_mode, parent_graph=self.graph)

    def apply_depolarizing_noise(self):
        """Depolarizing noise: randomly apply X/Y/Z with probability noise_prob."""
        if self.noise_prob <= 0:
            return
        for i in range(self.n_wires):
            if torch.rand(1).item() < self.noise_prob:
                err = torch.randint(0, 3, (1,)).item()
                if err == 0:
                    tqf.x(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                elif err == 1:
                    tqf.y(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)
                else:
                    tqf.z(self.q_device, wires=i, static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(self,
                x: torch.Tensor,
                q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        Forward pass: random features → VQC → measurement → HyperNetwork → weight matrix W.

        Args:
            x:         Input random features [batch_size, feature_dim]
            q_device:  Quantum device (pre-initialized with batch size)
        Returns:
            W: Generated weight matrix [out_dim, in_dim]
        """
        self.q_device = q_device
        bsz = x.shape[0]

        # Reset quantum state
        self.q_device.reset_states(bsz)

        # Step 1: Encode classical features into quantum state
        self.encoder(self.q_device, x)

        # Step 2: Variational quantum layers + entanglement + noise
        for k in range(self.n_qlayers):
            self.apply_variational_layer(k)
            self.apply_entanglement()
            self.apply_depolarizing_noise()

        # Step 3: Quantum measurement → [bsz, n_wires]
        q_out = self.measure(self.q_device).to(x.device)

        # Step 4: HyperNetwork generates weight factors
        latent = self.hyper_net(q_out)                        # [bsz, latent_dim]
        U_flat = self.fc_U(latent)                            # [bsz, out_dim * rank]
        V_flat = self.fc_V(latent)                            # [bsz, in_dim * rank]

        U = U_flat.view(-1, self.out_dim, self.rank)          # [bsz, out_dim, rank]
        V = V_flat.view(-1, self.in_dim, self.rank)           # [bsz, in_dim, rank]

        # Step 5: Combine weight matrix W = U @ V^T
        W = torch.bmm(U, V.transpose(1, 2))                   # [bsz, out_dim, in_dim]

        # Average pool to a single matrix (or keep batch dim as needed)
        W = W.mean(dim=0)                                     # [out_dim, in_dim]
        # Stabilize training: scale weights
        W = W / (self.rank ** 0.5)

        self._cached_W = W.detach()
        return W

    def get_generated_weight(self) -> torch.Tensor:
        """Return the most recently generated weight matrix."""
        if self._cached_W is None:
            raise RuntimeError("Run forward() first to generate weights")
        return self._cached_W


class VQC_MLPNet(nn.Module):
    """
    VQC-MLPNet: End-to-end network with VQC weight generation + MLP inference.

    Training / inference flow:
      1. VQCWeightGenerator generates weight W from random features
      2. Use W as the Linear layer weight to transform input data
      3. Follow with additional layers for classification / regression tasks
    """

    def __init__(self,
                 n_wires: int = 12,
                 n_qlayers: int = 3,
                 weight_shape: tuple = (4304, 1152),
                 rank: int = 64,
                 latent_dim: int = 256,
                 out_features: int = 2,
                 noise_prob: float = 0.0):
        super().__init__()
        self.weight_shape = weight_shape
        out_dim, in_dim = weight_shape

        # VQC weight generator
        self.weight_gen = VQCWeightGenerator(
            n_wires=n_wires,
            n_qlayers=n_qlayers,
            weight_shape=weight_shape,
            rank=rank,
            latent_dim=latent_dim,
            noise_prob=noise_prob,
        )

        # Final classification layer
        self.fc_out = nn.Linear(out_dim, out_features)

    def forward(self,
                data: torch.Tensor,
                random_features: torch.Tensor,
                q_device: tq.QuantumDevice) -> torch.Tensor:
        """
        Args:
            data:             Actual input data [batch, in_dim]
            random_features:  Random features fed into VQC [batch, feature_dim]
            q_device:         Quantum device
        Returns:
            output: [batch, out_features]
        """
        # Step 1: VQC generates weight matrix W [out_dim, in_dim]
        W = self.weight_gen(random_features, q_device)

        # Step 2: Use generated weights for linear transform: y = data @ W^T
        W = W.to(data.device)
        hidden = F.linear(data, W)            # [batch, out_dim]
        hidden = F.relu(hidden)

        # Step 3: Final classification layer
        output = self.fc_out(hidden)
        return output

    def generate_weights_only(self,
                              random_features: torch.Tensor,
                              q_device: tq.QuantumDevice) -> torch.Tensor:
        """Generate weight matrix only, without MLP inference."""
        return self.weight_gen(random_features, q_device)


class VQCGeneratedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with weights dynamically generated by VQC.

    Instead of storing a fixed weight matrix, this layer holds a learnable
    *context vector* that is fed through a VQC → HyperNetwork pipeline to
    produce the weight matrix at each forward pass. Gradients flow through
    the VQC simulator, enabling end-to-end quantum-classical training.

    Usage:
        layer = VQCGeneratedLinear(in_features=512, out_features=128,
                                    n_wires=10, n_qlayers=2, rank=32)
        output = layer(input)   # same interface as nn.Linear
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_wires: int = 10,
                 n_qlayers: int = 2,
                 rank: int = 32,
                 latent_dim: int = 128,
                 noise_prob: float = 0.0,
                 bias: bool = True):
        """
        Args:
            in_features:  Input feature dimension (nn.Linear convention)
            out_features: Output feature dimension
            n_wires:      Number of qubits
            n_qlayers:    VQC variational layers
            rank:         Low-rank decomposition rank
            latent_dim:   HyperNetwork hidden dimension
            noise_prob:   Depolarizing noise probability
            bias:         Whether to include a bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_wires = n_wires
        feature_dim = 2 ** n_wires  # AmplitudeEncoder requirement

        # ---- Learnable context vector (replaces random features) ----
        self.context = nn.Parameter(torch.randn(1, feature_dim) * 0.02)

        # ---- VQC weight generator (produces [out_features, in_features]) ----
        self.weight_gen = VQCWeightGenerator(
            n_wires=n_wires,
            n_qlayers=n_qlayers,
            weight_shape=(out_features, in_features),
            rank=rank,
            latent_dim=latent_dim,
            noise_prob=noise_prob,
        )

        # ---- Bias (same as nn.Linear) ----
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., in_features]
        Returns:
            output: [..., out_features]
        """
        # Generate weight matrix from learnable context
        q_dev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=1, device=x.device)
        W = self.weight_gen(self.context, q_dev)   # [out_features, in_features]
        return F.linear(x, W, self.bias)
