"""Multi-scale spectral branch (TimesBlock-lite).

Given input ``x ∈ R^{B×T×C}``, we identify the ``top-k`` global periods via
the FFT amplitude spectrum (averaged across the batch and channels), reshape
the signal into a 2-D period-vs-intra-period grid, and run a small 2-D
convolution over the grid. The branch returns one ``d_model``-dim feature
per channel that summarises periodic structure complementary to the
temporal-attention branch.

This is a deliberately stripped-down version of TimesNet's TimesBlock —
sufficient as an inductive bias and orders-of-magnitude cheaper.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fft_topk_periods(x: torch.Tensor, k: int) -> List[int]:
    """Return ``k`` candidate periods (in timesteps) by averaging the FFT
    amplitude across the batch and channel dims and picking the ``k`` largest
    non-DC bins.
    """
    # x: (B, T, C). Cast to fp32 — fp16 FFT triggers ComplexHalf warnings.
    amp = torch.fft.rfft(x.float(), dim=1).abs()  # (B, T//2+1, C)
    amp = amp.mean(dim=(0, 2))            # (T//2+1,)
    amp[0] = 0.0
    T = x.shape[1]
    bins = torch.topk(amp, k=min(k, amp.numel() - 1)).indices.tolist()
    periods = []
    for b in bins:
        if b == 0:
            continue
        p = max(2, T // b)
        if p not in periods:
            periods.append(p)
    if not periods:
        periods = [max(2, T // 2)]
    return periods


class _PeriodConv(nn.Module):
    """One 2-D conv block that operates on a period × intra-period grid."""

    def __init__(self, channels: int, hidden: int, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=kernel, padding=kernel // 2),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)


class SpectralBranch(nn.Module):
    """Top-k period decomposition with a shared 2-D conv.

    Parameters
    ----------
    n_channels : int  (C — number of input variates)
    d_model : int     output feature dim per channel
    window : int      training window length T
    topk : int        number of dominant periods
    hidden : int      conv hidden width
    """

    def __init__(self, n_channels: int, d_model: int, window: int, topk: int = 3, hidden: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.window = window
        self.topk = topk
        self.conv = _PeriodConv(channels=n_channels, hidden=hidden)
        # Per-channel projection from time-series back to a d_model summary
        self.proj = nn.Linear(window, d_model)

    def _fold(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """Reshape (B, T, C) → (B, C, n_periods, period) padding T to a multiple
        of ``period`` with zeros."""
        B, T, C = x.shape
        pad = (period - (T % period)) % period
        if pad:
            x = F.pad(x, (0, 0, 0, pad))  # pad along T axis (dim=1 of x)
        n_periods = (T + pad) // period
        # (B, T+pad, C) → (B, C, n_periods, period)
        grid = x.permute(0, 2, 1).reshape(B, C, n_periods, period)
        return grid, pad

    def _unfold(self, grid: torch.Tensor, pad: int) -> torch.Tensor:
        """(B, C, n_periods, period) → (B, T, C)."""
        B, C, n_periods, period = grid.shape
        x = grid.reshape(B, C, n_periods * period).permute(0, 2, 1)
        if pad:
            x = x[:, :-pad, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-channel features ``(B, C, d_model)``."""
        with torch.no_grad():
            periods = _fft_topk_periods(x.detach(), self.topk)
        outs = []
        weights = []
        for p in periods:
            grid, pad = self._fold(x, p)              # (B, C, n_periods, p)
            feat = self.conv(grid)                    # (B, C, n_periods, p)
            seq = self._unfold(feat, pad)             # (B, T, C)
            outs.append(seq)
            # weight by amplitude proxy: variance of the folded grid mean
            weights.append(grid.pow(2).mean(dim=(2, 3)).mean(dim=1, keepdim=True))
        # weighted sum across periods
        stacked = torch.stack(outs, dim=0)            # (k, B, T, C)
        w = torch.stack(weights, dim=0)               # (k, B, 1)
        w = F.softmax(w, dim=0).unsqueeze(-1).unsqueeze(-1)  # (k, B, 1, 1, 1)
        # broadcast weights — treat per-period feature as residual contribution
        merged = (stacked * w.squeeze(-1)).sum(dim=0)  # (B, T, C)
        # per-channel summary via linear projection over T
        return self.proj(merged.permute(0, 2, 1).contiguous())  # (B, C, d_model)
