"""Composite training loss for CCG-MSD.

Components
----------
``L_rec``  — masked reconstruction MSE; we randomly zero ``mask_rate`` of
             timesteps in the input and weight the MSE on the masked
             positions ``mask_w``× higher than the rest.

``L_outl`` — synthetic-outlier reconstruction (AnomalyBERT-style). With
             probability ``outlier_rate`` we corrupt a small interval of
             the input with a heavy-tailed perturbation; the model is
             penalised for *failing* to reconstruct the *clean* original
             at those timesteps.

``L_freq`` — FFT-domain L1 on top-k bands. Useful inductive bias for
             periodic signals.

``L_nll``  — Gaussian NLL on the density-head ``(μ, σ²)``. Optional;
             enabled when ``nll_w > 0``.

``L_dag``  — acyclicity penalty on the channel adjacency, scheduled
             externally and multiplied in by the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    rec: float = 1.0
    nll: float = 0.0
    freq: float = 0.3
    outlier: float = 1.0
    mask_rate: float = 0.15
    mask_w: float = 5.0
    outlier_rate: float = 0.10
    outlier_strength: float = 4.0
    freq_topk: int = 8


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def random_mask(
    x: torch.Tensor, mask_rate: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero ``mask_rate`` fraction of timesteps; return masked input and
    boolean mask ``(B, T, 1)`` over the masked positions."""
    B, T, C = x.shape
    mask = torch.rand(B, T, 1, device=x.device) < mask_rate
    masked = x.masked_fill(mask, 0.0)
    return masked, mask


def synthetic_outliers(
    x: torch.Tensor, rate: float, strength: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace short intervals with heavy-tailed noise. Returns the corrupted
    input and a boolean mask ``(B, T, 1)`` of the corrupted positions."""
    if rate <= 0:
        return x.clone(), torch.zeros(x.shape[0], x.shape[1], 1, dtype=torch.bool, device=x.device)
    B, T, C = x.shape
    out = x.clone()
    mask = torch.zeros(B, T, 1, dtype=torch.bool, device=x.device)
    n_intervals = int(rate * T / 8)  # average interval length 8
    if n_intervals <= 0:
        return out, mask
    for b in range(B):
        for _ in range(n_intervals):
            length = int(torch.randint(low=4, high=12, size=(1,), device=x.device).item())
            start = int(torch.randint(low=0, high=max(1, T - length), size=(1,), device=x.device).item())
            noise = torch.randn(length, C, device=x.device) * strength
            out[b, start : start + length] = noise
            mask[b, start : start + length, 0] = True
    return out, mask


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


def reconstruction_loss(
    x_clean: torch.Tensor, recon: torch.Tensor, mask: Optional[torch.Tensor], mask_w: float
) -> torch.Tensor:
    err = (x_clean - recon).pow(2)
    if mask is None:
        return err.mean()
    weighted = err * (1.0 + (mask_w - 1.0) * mask.float())
    return weighted.mean()


def gaussian_nll(
    x_clean: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return (0.5 * (logvar + (x_clean - mu).pow(2) * inv_var)).mean()


def freq_loss(x_clean: torch.Tensor, recon: torch.Tensor, topk: int) -> torch.Tensor:
    # FFT lacks a stable fp16 kernel; cast to fp32 inside the loss to silence
    # ComplexHalf warnings without disabling autocast in the rest of the model.
    fx = torch.fft.rfft(x_clean.float(), dim=1)
    fr = torch.fft.rfft(recon.float(), dim=1)
    diff = (fx - fr).abs()                           # (B, F, C)
    if topk and topk < diff.shape[1]:
        # Keep the top-k bands per channel by amplitude of x_clean
        amp = fx.abs().mean(dim=0)                   # (F, C)
        idx = amp.topk(topk, dim=0).indices          # (k, C)
        # Gather diff at those indices per channel
        # diff: (B, F, C) → permute to (B, C, F), gather along F
        diffp = diff.permute(0, 2, 1)                # (B, C, F)
        gathered = torch.gather(
            diffp, dim=2, index=idx.T.unsqueeze(0).expand(diffp.shape[0], -1, -1)
        )
        return gathered.mean()
    return diff.mean()


# ---------------------------------------------------------------------------
# Top-level callable
# ---------------------------------------------------------------------------


def composite_loss(
    model_out: Dict[str, torch.Tensor],
    x_clean: torch.Tensor,
    x_input: torch.Tensor,
    mask: Optional[torch.Tensor],
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    recon = model_out["recon"]
    mu = model_out["mu"]
    logvar = model_out["logvar"]

    L_rec = reconstruction_loss(x_clean, recon, mask, weights.mask_w) * weights.rec

    components = {"rec": L_rec.detach(), "rec_w": L_rec}
    total = L_rec
    if weights.nll > 0:
        L_nll = gaussian_nll(x_clean, mu, logvar) * weights.nll
        components["nll"] = L_nll.detach()
        total = total + L_nll
    if weights.freq > 0:
        L_freq = freq_loss(x_clean, recon, weights.freq_topk) * weights.freq
        components["freq"] = L_freq.detach()
        total = total + L_freq

    components["total"] = total
    return components
