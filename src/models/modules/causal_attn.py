"""Causal Channel-Graph Attention + parameterisation of the channel adjacency.

The adjacency ``A ∈ R^{C×C}`` is parameterised in low-rank form::

    A = sigmoid(U V^T) ⊙ M_prior

with ``U, V ∈ R^{C×r}``. ``M_prior`` is a 0/1 (or soft) mask encoding any
exogenous prior — e.g. the UAV control→IMU partial order built by
``src.data.alfa.build_uav_prior_mask`` — and defaults to all-ones (fully
data-driven).

Acyclicity is enforced via the ``dag_loss`` module; this layer simply gates
softmax attention weights by ``log A`` so an entry of ``A`` near zero
suppresses that channel-pair's attention.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelDAG(nn.Module):
    """Low-rank, optionally prior-masked channel adjacency."""

    def __init__(
        self,
        n_channels: int,
        rank: int = 16,
        prior_mask: Optional[torch.Tensor] = None,
        zero_diagonal: bool = True,
        init_corr: Optional[torch.Tensor] = None,
        init_logit_bias: float = -3.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.zero_diagonal = zero_diagonal
        self.U = nn.Parameter(torch.empty(n_channels, rank))
        self.V = nn.Parameter(torch.empty(n_channels, rank))
        # Sparse-by-default initialisation: shift logits to make
        # sigmoid output ~0.05 so h(A) starts ~ 0 and the augmented
        # Lagrangian schedule has a meaningful starting point. Without
        # this, dense random A overwhelms the reconstruction signal.
        self.bias = nn.Parameter(torch.full((1,), float(init_logit_bias)))
        nn.init.normal_(self.U, std=0.01)
        nn.init.normal_(self.V, std=0.01)
        if init_corr is not None:
            # SVD-based warm start: set U,V so sigmoid(U V^T) ≈ init_corr
            # Use logit on a soft target.
            target = torch.logit(init_corr.clamp(0.05, 0.95))
            U_, S_, V_ = torch.linalg.svd(target, full_matrices=False)
            r = min(rank, S_.shape[0])
            with torch.no_grad():
                self.U.copy_(U_[:, :r] * S_[:r].sqrt())
                self.V.copy_(V_[:r].T * S_[:r].sqrt())
        if prior_mask is not None:
            self.register_buffer("prior_mask", prior_mask.float())
        else:
            self.prior_mask = None

    def forward(self) -> torch.Tensor:
        adj = torch.sigmoid(self.U @ self.V.T + self.bias)  # (C, C)
        if self.zero_diagonal:
            # diagonal handled separately by self-attention path
            mask = 1.0 - torch.eye(self.n_channels, device=adj.device, dtype=adj.dtype)
            adj = adj * mask
        if self.prior_mask is not None:
            adj = adj * self.prior_mask
        return adj


class ChannelGraphAttention(nn.Module):
    """Multi-head channel attention masked by a learned adjacency.

    Input  ``h`` : ``(B, C, d_model)`` — one token per channel.
    Adjacency ``adj`` : ``(C, C)`` non-negative, 0 in suppressed positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, D = h.shape
        qkv = self.qkv(h).reshape(B, C, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, C, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, C, C)
        if adj is not None:
            # Convert multiplicative gate to additive log-bias for numerical
            # stability after softmax. Adj entries near 0 → very negative bias.
            log_adj = torch.log(adj.clamp(min=1e-6))
            attn = attn + log_adj.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, C, D)
        return self.proj_drop(self.out(out))
