"""Interchangeable recurrent global memories and a shared-parameter speaker GRU.

GatedDeltaMemory is a portable utterance-level adaptation of the gated delta
rule (Yang et al., 2025, Eq. 8, https://arxiv.org/abs/2412.06464). It implements
the exact matrix recurrence, not the paper's complete language-model block:
there is no short convolution, stacked SwiGLU network or fused CUDA kernel.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MemoryState:
    summary: torch.Tensor
    matrix: torch.Tensor | None = None

    def keep_inactive(self, previous: "MemoryState", active: torch.Tensor) -> "MemoryState":
        matrix = self.matrix
        if matrix is not None and previous.matrix is not None:
            matrix = torch.where(active[:, None, None, None], matrix, previous.matrix)
        return MemoryState(torch.where(active[:, None], self.summary, previous.summary), matrix)


class GRUMemory(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)

    def initial_state(self, reference: torch.Tensor) -> MemoryState:
        return MemoryState(torch.zeros_like(reference))

    def forward(self, inputs: torch.Tensor, state: MemoryState) -> MemoryState:  # ty: ignore[invalid-mutable-override]
        return MemoryState(self.cell(inputs, state.summary))


class GatedDeltaMemory(nn.Module):
    """Multi-head associative memory, with FP32 accumulation under autocast.

    For each head: S' = alpha*S; S_new = S' + beta*(v - S'@k)@k.T.
    The readout is S_new@q. Queries and keys are L2 normalized.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.decay = nn.Linear(hidden_dim, num_heads)
        self.write = nn.Linear(hidden_dim, num_heads)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.constant_(self.decay.bias, 2.0)

    def initial_state(self, reference: torch.Tensor) -> MemoryState:
        shape = (reference.shape[0], self.num_heads, self.head_dim, self.head_dim)
        matrix = torch.zeros(shape, device=reference.device, dtype=torch.float32)
        return MemoryState(torch.zeros_like(reference), matrix)

    def forward(self, inputs: torch.Tensor, state: MemoryState) -> MemoryState:  # ty: ignore[invalid-mutable-override]
        if state.matrix is None:
            raise ValueError("Gated delta memory requires a matrix state")
        normalized = self.input_norm(inputs)
        q, k, v = (
            self.qkv(normalized)
            .reshape(inputs.shape[0], 3, self.num_heads, self.head_dim)
            .unbind(1)
        )
        alpha = self.decay(normalized).float().sigmoid()
        beta = self.write(normalized).float().sigmoid()
        # Disabling autocast here prevents matmuls from downcasting the accumulator.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            decayed = state.matrix.float() * alpha[..., None, None]
            old_value = (decayed @ k.unsqueeze(-1)).squeeze(-1)
            residual = beta[..., None] * (v.float() - old_value)
            matrix = decayed + residual.unsqueeze(-1) * k.unsqueeze(-2)
            readout = (matrix @ q.unsqueeze(-1)).squeeze(-1).flatten(1)
        gated = readout.to(inputs.dtype) * F.silu(self.output_gate(normalized))
        summary = self.output_norm(inputs + self.output(gated))
        return MemoryState(summary, matrix)


class SpeakerMemory(nn.Module):
    """One set of GRU parameters; the caller maintains one state per speaker."""

    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.input = nn.Linear(3 * hidden_dim + time_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, utterance, interaction, global_summary, time, previous):  # ty: ignore[invalid-mutable-override]
        inputs = torch.cat((utterance, interaction, global_summary, time), dim=-1)
        return self.cell(torch.tanh(self.input(inputs)), previous)
