"""Differentiable acyclicity penalty for the channel-graph adjacency.

We follow Zheng et al. 2018 (NOTEARS):

    h(A) = tr(exp(A ∘ A)) - C

with ``C = A.shape[0]``, where ``∘`` is element-wise product. ``h(A) ≥ 0``
and ``h(A) = 0`` iff A is acyclic. Squaring gives a smooth penalty.

Numerical notes
---------------
``torch.linalg.matrix_exp`` works for C up to a few hundred on V100. For
larger C we provide a truncated-power-series alternative
``h_polynomial(A, k)`` (Yu et al. 2019) that avoids the eigendecomposition.

We also expose an **augmented-Lagrangian schedule** helper that linearly
ramps the penalty weight from 0 → ``lambda_max`` over the first
``warmup_epochs``. This is the simplified schedule documented in the plan;
it is sufficient for end-to-end deep-learning training because the model's
reconstruction objective already biases the adjacency away from
fully-saturated solutions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def h_expm(adj: Tensor, normalize: bool = True) -> Tensor:
    """NOTEARS acyclicity score using matrix exponential.

    Parameters
    ----------
    adj : (C, C) tensor
        Channel adjacency, expected non-negative (e.g., output of sigmoid).
    normalize : bool
        If True (default), divide by ``C`` so the penalty is bounded
        independent of channel count. The published NOTEARS paper uses
        the unnormalised form, but for our deep-learning training where
        the penalty is added to a per-sample reconstruction loss the
        normalised form yields a stable ``λ`` schedule across datasets.

    Notes
    -----
    ``torch.linalg.matrix_exp`` lacks a fp16 implementation, so we cast
    the input to fp32 internally. For dense adjacency at random init,
    ``trace(exp(A∘A)) - C`` can reach ``C·exp(C·E[A²])`` which exceeds
    1e6 for C ≥ 30 — completely overwhelming the reconstruction signal.
    Normalisation by ``C`` (or by squaring once normalised) is essential
    for end-to-end DL training; the augmented-Lagrangian schedule then
    ramps a small ``λ`` cleanly.
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square 2D; got {adj.shape}")
    C = adj.shape[0]
    adj32 = adj.float()
    # Clamp the elementwise square so exp(M) is well-behaved at random init.
    # Acyclicity is preserved by any monotone transform.
    M = (adj32 * adj32).clamp(max=1.0)
    expm = torch.linalg.matrix_exp(M)
    h = torch.diagonal(expm).sum() - C
    return h / C if normalize else h


def h_polynomial(adj: Tensor, k: int = 8, alpha: float = None, normalize: bool = True) -> Tensor:
    """Polynomial-series acyclicity (Yu et al. 2019), for large C.

    h(A) = tr((I + α·A∘A)^k) - C, with α = 1/C by default.

    Equivalent to NOTEARS as ``k → ∞``; for k≈C/2 it is a tight surrogate.
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square 2D; got {adj.shape}")
    C = adj.shape[0]
    if alpha is None:
        alpha = 1.0 / max(C, 1)
    adj32 = adj.float()
    eye = torch.eye(C, device=adj32.device, dtype=adj32.dtype)
    M = eye + alpha * adj32 * adj32
    Mk = torch.linalg.matrix_power(M, k)
    h = torch.diagonal(Mk).sum() - C
    return h / C if normalize else h


def dag_penalty(
    adj: Tensor, mode: str = "expm", k: int = 8, square: bool = True
) -> Tensor:
    """Convenience wrapper. ``square=True`` (default) returns ``h(A)²``."""
    if mode == "expm":
        h = h_expm(adj)
    elif mode == "poly":
        h = h_polynomial(adj, k=k)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return h * h if square else h


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass
class DAGSchedule:
    """Linear warm-up of the acyclicity penalty weight.

    Used as::

        sched = DAGSchedule(warmup_epochs=10, lambda_max=0.1)
        for epoch in ...:
            lam = sched(epoch)
            loss = ... + lam * dag_penalty(A)
    """

    warmup_epochs: int = 10
    lambda_max: float = 0.1
    schedule: str = "linear"   # or "cosine"

    def __call__(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return float(self.lambda_max)
        frac = min(1.0, max(0.0, epoch / self.warmup_epochs))
        if self.schedule == "linear":
            return float(self.lambda_max * frac)
        if self.schedule == "cosine":
            return float(self.lambda_max * 0.5 * (1.0 - math.cos(math.pi * frac)))
        raise ValueError(f"unknown schedule {self.schedule!r}")


@dataclass
class PhysSchedule:
    """Cosine decay of the physics-prior consistency weight from
    ``phys_w_max`` to ``phys_w_min`` over ``decay_epochs`` epochs, then
    held flat at ``phys_w_min``.

    Motivation: pilot 1 showed the prior is informative early (acts as a
    structural inductive bias and improves epoch-1/2 fault-unseen
    generalisation) but counterproductive late (over-constrains the
    adjacency once enough data-driven signal is available). A monotonic
    decay lets the prior dominate at init and yield to the data later.
    """

    decay_epochs: int = 0
    phys_w_max: float = 0.0
    phys_w_min: float = 0.0

    def __call__(self, epoch: int) -> float:
        if self.decay_epochs <= 0:
            return float(self.phys_w_max)
        frac = min(1.0, max(0.0, epoch / self.decay_epochs))
        decay = 0.5 * (1.0 + math.cos(math.pi * frac))   # 1 -> 0
        return float(self.phys_w_min + (self.phys_w_max - self.phys_w_min) * decay)


# ---------------------------------------------------------------------------
# Initialisation helper: correlation-thresholded warm start
# ---------------------------------------------------------------------------


@torch.no_grad()
def correlation_init(x: Tensor, threshold: float = 0.3) -> Tensor:
    """Compute ``|corr(channels)|`` and threshold to give a sparse seed
    adjacency. Diagonal is zeroed (no self-loops in the prior).

    Parameters
    ----------
    x : (N, T, D) or (T, D) tensor of *training* observations.
    threshold : float
        Edges with absolute Pearson correlation below this are zeroed.
    """
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.dim() != 2:
        raise ValueError(f"x must be 2D or 3D; got {tuple(x.shape)}")
    x = x - x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True) + 1e-6
    x = x / sd
    C = x.shape[1]
    corr = (x.T @ x) / max(1, x.shape[0] - 1)
    corr = corr.abs()
    corr = torch.where(corr >= threshold, corr, torch.zeros_like(corr))
    corr.fill_diagonal_(0.0)
    return corr.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    C = 8
    # exact DAG → upper triangular
    A_dag = torch.triu(torch.rand(C, C), diagonal=1)
    A_cyc = A_dag.clone()
    A_cyc[3, 1] = 0.5  # introduce a cycle
    print("h(DAG) =", h_expm(A_dag).item(), "≈ 0")
    print("h(cyc) =", h_expm(A_cyc).item(), "> 0")
    print("h_poly(DAG) =", h_polynomial(A_dag, k=8).item())
    s = DAGSchedule(warmup_epochs=10, lambda_max=0.1)
    print("schedule:", [s(e) for e in range(0, 12)])
