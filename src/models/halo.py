"""HALO: Hierarchical Anomaly via Layered Operations for UAV telemetry.

HALO is a layer-causal cross-attention reconstruction detector. It is
explicitly designed around the UAV control architecture rather than
treating the multivariate stream as a homogeneous channel set.

Architecture
------------
The 16 ALFA channels are partitioned by the UAV control architecture
into 5 layers (command, servo, sensor, IMU, state). Each layer hosts
its own transformer encoder over the channels in that layer; a
forward-direction cross-attention block lets a downstream layer attend
to all upstream layers' encoded representations. The reconstruction
target is each layer's own channels, predicted from the cross-attended
upstream context. A per-layer reconstruction MSE is the anomaly score;
the scores from different layers are summed with per-layer weights
derived from inverse channel count so deeper-but-fewer state channels
do not dominate.

  L0 command  --(self-attn)-->  enc0  ----.
  L1 servo    --(self-attn)-->  enc1 <--+--cross--> dec1 -> recon servo
  L2 sensor   --(self-attn)-->  enc2 <--+--cross--> dec2 -> recon sensor
  L3 imu      --(self-attn)-->  enc3 <--+--cross--> dec3 -> recon imu
  L4 state    --(self-attn)-->  enc4 <--+--cross--> dec4 -> recon state

Why this is novel
-----------------
- No learned channel adjacency. The layer assignment is exogenous prior
  knowledge about the UAV control architecture; no NOTEARS, no DAG loss.
- No homogeneous channel-as-token transformer either; each layer has
  its own encoder, so the model cannot mix command and state semantics
  in the encoder.
- Cross-attention is unidirectional (forward in the layer order) so the
  model literally cannot use state to reconstruct command, which a
  homogeneous transformer would silently allow.
- The detector is naturally robust to missing actuator channels: the
  cross-attention key/value tensor at the upstream layer just becomes
  shorter, and the downstream decoder still emits a state estimate.

Forward output
--------------
``recon_by_layer`` : dict[int, (B, T, C_l)]   per-layer reconstruction
``recon``          : (B, T, C)   concatenated reconstruction in layer order
``layer_index``    : list[int]   per-channel layer id (for diagnostics)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .pg_cgad import UAV_LAYERS, assign_layers


# ---------------------------------------------------------------------------
# Channel grouping by layer
# ---------------------------------------------------------------------------


def channel_partition(channel_names: List[str]) -> Tuple[List[List[int]], List[int]]:
    """Return ``(channels_in_layer, channel_layer_id)``.

    ``channels_in_layer[l]`` is the list of channel indices assigned to
    layer ``l``; ``channel_layer_id[c]`` is the layer id of channel ``c``.
    """
    layer_id = assign_layers(channel_names)
    n_layers = max(layer_id) + 1 if layer_id else 0
    by_layer: List[List[int]] = [[] for _ in range(n_layers)]
    for c, L in enumerate(layer_id):
        by_layer[L].append(c)
    return by_layer, layer_id


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MHA(nn.Module):
    """Standard multi-head attention; supports self- and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = q_in.shape
        Nk = kv_in.shape[1]
        q = self.q(q_in).reshape(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(kv_in).reshape(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(kv_in).reshape(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        h = (attn @ v).transpose(1, 2).contiguous().reshape(B, Nq, -1)
        return self.out(h)


class LayerEncoder(nn.Module):
    """Self-attention transformer over the channels of one UAV layer.

    Input shape: ``(B, n_layer_channels, d_model)`` -- one token per
    channel in this layer, embedding dimension ``d_model``.
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "attn": _MHA(d_model, n_heads, dropout),
                "ffn": _FFN(d_model, dropout),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
            }))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h = h + blk["attn"](blk["ln1"](h), blk["ln1"](h))
            h = h + blk["ffn"](blk["ln2"](h))
        return self.norm(h)


class CrossLayerDecoder(nn.Module):
    """Cross-attention block: queries this layer, keys/values from upstream
    concatenated context."""

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "cross": _MHA(d_model, n_heads, dropout),
                "ffn": _FFN(d_model, dropout),
                "ln_q": nn.LayerNorm(d_model),
                "ln_kv": nn.LayerNorm(d_model),
                "ln_ff": nn.LayerNorm(d_model),
            }))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            q = q + blk["cross"](blk["ln_q"](q), blk["ln_kv"](kv))
            q = q + blk["ffn"](blk["ln_ff"](q))
        return self.norm(q)


# ---------------------------------------------------------------------------
# v5: multi-period block (TimesNet-inspired)
# ---------------------------------------------------------------------------


class MultiPeriodBlock(nn.Module):
    """For each channel, decompose the window into top-K periodic 2D views
    via FFT and apply small 2D conv. Captures multi-scale temporal patterns
    the channel-as-token Linear(T->d_model) projection misses.

    v6: ``kernels`` controls a multi-receptive parallel-conv branch
    (e.g. [3, 7, 15] for fast/mid/slow fault dynamics). Each kernel size
    runs its own conv branch; the per-kernel features are averaged before
    out_proj. Backward-compatible single-kernel behaviour comes through
    ``kernels=[k]``.

    Input:  (B, T, C)
    Output: (B, C, d_model) -- a per-channel embedding to add to the
            iTransformer-style channel embedding.
    """

    def __init__(self, window: int, d_model: int, top_k: int = 3,
                 inner_dim: int = 32, kernel: int = 3,
                 kernels: Optional[List[int]] = None):
        super().__init__()
        self.window = int(window)
        self.top_k = int(top_k)
        self.inner_dim = int(inner_dim)
        if kernels is None:
            kernels = [kernel]
        self.kernels = list(kernels)

        # Per-kernel parallel conv branch: each runs Conv2d(1->inner) ->
        # GELU -> Conv2d(inner->inner) with same-padding for its own kernel.
        # Inputs are (B*C, 1, P, n_p); outputs (B*C, inner). Branches are
        # averaged before projection.
        self.branches = nn.ModuleList()
        for k in self.kernels:
            pad = k // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(1, inner_dim, kernel_size=k, padding=pad),
                nn.GELU(),
                nn.Conv2d(inner_dim, inner_dim, kernel_size=k, padding=pad),
            ))
        self.out_proj = nn.Linear(inner_dim, d_model)

    @staticmethod
    def _fft_topk_periods(x_fp32: torch.Tensor, k: int) -> List[int]:
        """Pick top-K periods using mean FFT amplitude across B and C.
        Returns a list of period lengths (>=2). DC bin is masked out.
        """
        B, T, C = x_fp32.shape
        xf = torch.fft.rfft(x_fp32, dim=1)
        amp = xf.abs().mean(dim=(0, 2))
        amp[0] = 0.0
        topk = torch.topk(amp, min(k, amp.numel() - 1)).indices
        periods: List[int] = []
        for f in topk.tolist():
            if f <= 0:
                continue
            P = max(2, T // f)
            P = min(P, T)
            periods.append(P)
        return periods or [max(2, T // 4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # FFT in fp32 for numerical stability (autocast-safe)
        with torch.cuda.amp.autocast(enabled=False):
            periods = self._fft_topk_periods(x.detach().float(), self.top_k)

        feats: List[torch.Tensor] = []
        for P in periods:
            n_p = T // P
            if n_p < 1:
                continue
            x_p = x[:, : n_p * P, :]                        # (B, n_p*P, C)
            x_p = x_p.reshape(B, n_p, P, C)
            x_p = x_p.permute(0, 3, 2, 1).contiguous()      # (B, C, P, n_p)
            x_p = x_p.reshape(B * C, 1, P, n_p)
            # average across multi-kernel branches
            branch_outs = [br(x_p) for br in self.branches]
            f = torch.stack(branch_outs, dim=0).mean(0)     # (B*C, inner, P, n_p)
            f = f.mean(dim=(2, 3))                           # (B*C, inner)
            feats.append(f.reshape(B, C, self.inner_dim))
        if not feats:
            return x.new_zeros(B, C, self.out_proj.out_features)
        avg = torch.stack(feats, dim=0).mean(0)              # (B, C, inner)
        return self.out_proj(avg)                            # (B, C, d_model)


# ---------------------------------------------------------------------------
# v12: TN-style parallel encoder branch (stacked SpectralBranch + residual)
# ---------------------------------------------------------------------------


class TNBranch(nn.Module):
    """Stacked SpectralBranch layers with a TimesNet-style residual stream.

    HALO v5's multi-period block only injects period-domain context ONCE at
    the input embedding; the rest of the encoder reasons in iTransformer
    channel-as-token attention. On LOFO splits (stuck-actuator faults) this
    leaves a gap vs TimesNet whose ENTIRE encoder runs in period-domain.

    v12 fixes that by adding a parallel path that mirrors the TimesNet
    backbone: ``n_layers`` SpectralBranch blocks each running on a ``(B, T, C)``
    residual stream, returning the final block's per-channel feature
    ``(B, C, d_model)`` which is added (gated) to HALO's channel embedding
    before the per-layer encoder. This gives HALO TN's deep period bias
    while keeping its layered cross-attention decoder intact.
    """

    def __init__(self, n_channels: int, d_model: int, window: int,
                 n_layers: int = 3, topk: int = 3, hidden: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        from .modules.spectral import SpectralBranch  # local import
        self.n_layers = int(n_layers)
        self.window = int(window)
        self.blocks = nn.ModuleList([
            SpectralBranch(n_channels=n_channels, d_model=d_model,
                           window=window, topk=topk, hidden=hidden)
            for _ in range(self.n_layers)
        ])
        self.expand = nn.Linear(d_model, window)
        self.norm = nn.LayerNorm(window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Maintain a (B, C, T) residual stream like TN baseline.
        h = x.permute(0, 2, 1).contiguous()                  # (B, C, T)
        last_feat: Optional[torch.Tensor] = None
        for block in self.blocks:
            feat = block(h.permute(0, 2, 1))                 # (B, C, d_model)
            last_feat = feat
            seq = self.expand(self.dropout(feat))            # (B, C, T)
            h = self.norm(h + seq)
        return last_feat                                      # (B, C, d_model)


class LSTMForecastHead(nn.Module):
    """v19: parallel LSTM forecast head. Takes the first (window - horizon)
    timesteps as context, autoregressively forecasts the last `horizon`
    timesteps. The forecast error (per-window MSE) is a multi-view anomaly
    signal that complements HALO's recon residual — engine-type LOFO faults
    cause continuous decline that breaks the LSTM's "continue normal" prior.

    Trained jointly with HALO via an auxiliary forecast loss; its score
    contribution is added to the recon score with weight ``lstm_score_weight``.
    """

    def __init__(self, n_channels: int, window: int, horizon: int = 20,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_channels = int(n_channels)
        self.window = int(window)
        self.horizon = int(horizon)
        self.context_len = self.window - self.horizon
        if self.context_len <= 0:
            raise ValueError("horizon must be < window")
        self.hidden = int(hidden)
        self.encoder = nn.LSTM(
            input_size=self.n_channels,
            hidden_size=self.hidden,
            num_layers=int(n_layers),
            dropout=float(dropout) if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(self.hidden, self.n_channels)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        ctx = x[:, :self.context_len, :]
        target = x[:, -self.horizon:, :]
        _, (h, c) = self.encoder(ctx)
        # Autoregressive forecast: feed last context step + step through horizon.
        last_step = ctx[:, -1:, :]
        forecasts = []
        for t in range(self.horizon):
            out, (h, c) = self.encoder(last_step, (h, c))
            pred = self.proj(out)            # (B, 1, C)
            forecasts.append(pred)
            last_step = pred                  # feed prediction back
        forecast = torch.cat(forecasts, dim=1)  # (B, horizon, C)
        return {"forecast": forecast, "target": target}

    @torch.no_grad()
    def anomaly_signal(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        err = (out["forecast"] - out["target"]).pow(2).mean(dim=(1, 2))  # (B,)
        return err

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return (out["forecast"] - out["target"]).pow(2).mean()


class HardInputStatGate(nn.Module):
    """v14: deterministic per-sample gate computed from input channel-std
    statistics. NO learnable parameters — gate value is fully determined by
    the relationship between this window's per-channel std and the training
    median std (registered as a buffer).

    For each channel: collapse_c = max(0, 1 - std(x)/median_train_c).
    Gate = max over active channels (median > min_active) of collapse_c.
    Stuck-actuator (LOFO) windows → gate ≈ 1. Heterogeneous (strat) windows
    → gate varies in [0,1].
    """

    def __init__(self, n_channels: int, min_active: float = 0.05):
        super().__init__()
        # placeholder; calibrated at first batch from train data via the
        # ``calibrate`` method below. Until then defaults to zero gate.
        self.register_buffer("median_c", torch.ones(n_channels))
        self.register_buffer("active", torch.zeros(n_channels))
        self.min_active = float(min_active)
        self._calibrated = False

    @torch.no_grad()
    def calibrate(self, train_window_stds: torch.Tensor):
        """train_window_stds: (N, C). Sets per-channel median + active mask."""
        med = torch.median(train_window_stds, dim=0).values
        self.median_c.copy_(med.to(self.median_c.device))
        self.active.copy_((med > self.min_active).float().to(self.active.device))
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Returns (B, 1, 1) in [0, 1].
        test_std = x.std(dim=1)  # (B, C)
        ratio = test_std / (self.median_c.unsqueeze(0) + 1e-6)
        collapse = (1.0 - ratio).clamp(min=0.0) * self.active.unsqueeze(0)
        gate = collapse.amax(dim=-1, keepdim=True)  # (B, 1)
        return gate.unsqueeze(-1)  # (B, 1, 1)


class DataDependentGate(nn.Module):
    """Per-sample sigmoid gate computed from input statistics.

    v12 used a single learnable scalar gate; with only 6 epochs the gate
    barely moved from init (≈0.11 final on strat). v13 instead computes
    one gate value per (B,) sample from input features: small MLP over a
    compact summary of x — per-channel std and dominant-period FFT power.
    Intuition: stuck-actuator inputs (LOFO) have low per-channel std and
    concentrated FFT power, distinguishable from strat's heterogeneous
    mix. The MLP learns to map low-variance/period-clean inputs to high
    gate (TN branch ON) and the rest to low gate.
    """

    def __init__(self, n_channels: int, hidden: int = 32, bias_init: float = -2.2):
        super().__init__()
        # Summary per sample: (per-channel std, per-channel peak-FFT-mag,
        # per-channel mean-abs) → 3*C-dim. MLP → 1.
        self.in_dim = 3 * int(n_channels)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Bias init so initial sigmoid(output) ≈ sigmoid(bias_init);
        # -2.2 → ~0.10 contribution like v12-B but per-sample learnable.
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, float(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Return (B, 1, 1) gate value to broadcast over (B, C, d_model).
        B, T, C = x.shape
        std = x.std(dim=1)                                     # (B, C)
        mabs = x.abs().mean(dim=1)                             # (B, C)
        with torch.cuda.amp.autocast(enabled=False):
            fft_mag = torch.fft.rfft(x.float(), dim=1).abs()   # (B, F, C)
        peak = fft_mag.amax(dim=1)                             # (B, C)
        feat = torch.cat([std, mabs, peak], dim=-1)            # (B, 3*C)
        # Normalise per sample for scale invariance
        feat = feat / (feat.amax(dim=-1, keepdim=True) + 1e-6)
        gate = torch.sigmoid(self.net(feat))                   # (B, 1)
        return gate.unsqueeze(-1)                               # (B, 1, 1)


# ---------------------------------------------------------------------------
# v9-B: TN-style 2D conv stem (channel × time)
# ---------------------------------------------------------------------------


class TNStem(nn.Module):
    """Multi-scale 2D conv over the (channel, time) lattice.

    HALO v5's per-channel embedding (Linear(window, d_model)) treats each
    channel's temporal trace in isolation and only mixes channels later via
    the cross-layer decoder. TimesNet beats HALO on LOFO splits by reasoning
    about cross-channel + multi-scale temporal motifs jointly in a 2D conv.
    v9-B borrows that bias: stem treats the input as a single 2D map of
    shape (B, 1, C, T), runs N parallel conv2d branches with different
    (kernel_c, kernel_t) sizes, sums them, time-pools, and projects back to
    a per-channel d_model embedding that the rest of HALO consumes.
    """

    def __init__(
        self,
        window: int,
        d_model: int,
        kernels: Optional[List[List[int]]] = None,
        inner_per_branch: int = 16,
    ):
        super().__init__()
        if kernels is None:
            kernels = [[3, 7], [3, 15], [5, 7]]
        self.kernels = [tuple(k) for k in kernels]
        self.branches = nn.ModuleList()
        for kc, kt in self.kernels:
            pad_c = kc // 2
            pad_t = kt // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(1, inner_per_branch, kernel_size=(kc, kt),
                          padding=(pad_c, pad_t)),
                nn.GELU(),
                nn.Conv2d(inner_per_branch, inner_per_branch,
                          kernel_size=(kc, kt), padding=(pad_c, pad_t)),
            ))
        total_inner = inner_per_branch * len(self.kernels)
        self.norm = nn.LayerNorm(total_inner)
        self.out_proj = nn.Linear(total_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Treat C × T as a 1-channel 2D image.
        B, T, C = x.shape
        x_2d = x.permute(0, 2, 1).unsqueeze(1)            # (B, 1, C, T)
        feats = [br(x_2d) for br in self.branches]
        h = torch.cat(feats, dim=1)                        # (B, S, C, T)
        h = h.mean(dim=-1)                                  # (B, S, C) — pool T
        h = h.permute(0, 2, 1)                              # (B, C, S)
        h = self.norm(h)
        return self.out_proj(h)                             # (B, C, d_model)


# ---------------------------------------------------------------------------
# HALO model
# ---------------------------------------------------------------------------


class HALO(nn.Module):
    def __init__(
        self,
        channel_names: List[str],
        window: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_blocks: int = 2,
        n_decoder_blocks: int = 2,
        dropout: float = 0.1,
        layer_dropout: float = 0.0,
        actuator_aug_rate: float = 0.0,
        actuator_aug_max_channels: int = 2,
        layer_score_weights: Optional[List[float]] = None,
        use_multi_period: bool = False,
        multi_period_topk: int = 3,
        multi_period_inner: int = 32,
        multi_period_kernels: Optional[List[int]] = None,
        multi_period_gate: bool = False,
        multi_period_gate_init: float = 0.0,
        pred_horizon: int = 0,
        pred_score_weight: float = 0.0,
        use_tn_stem: bool = False,
        tn_stem_kernels: Optional[List[List[int]]] = None,
        tn_stem_inner: int = 16,
        tn_stem_gate: bool = False,
        tn_stem_gate_init: float = 0.1,
        use_tn_branch: bool = False,
        tn_branch_layers: int = 3,
        tn_branch_topk: int = 3,
        tn_branch_hidden: int = 32,
        tn_branch_gate: bool = True,
        tn_branch_gate_init: float = 0.3,
        tn_branch_gate_data_dependent: bool = False,
        tn_branch_gate_mlp_hidden: int = 32,
        tn_branch_gate_bias_init: float = -2.2,
        tn_branch_gate_hard_input: bool = False,
        tn_branch_gate_hard_min_active: float = 0.05,
        tn_branch_gate_warmup_start: int = 0,
        tn_branch_gate_warmup_full: int = 0,
        use_tn_only_encoder: bool = False,
        tn_encoder_layers: int = 3,
        tn_encoder_topk: int = 3,
        tn_encoder_hidden: int = 64,
        use_lstm_forecast: bool = False,
        lstm_forecast_horizon: int = 20,
        lstm_forecast_hidden: int = 64,
        lstm_forecast_layers: int = 2,
        lstm_score_weight: float = 0.0,
        lstm_loss_weight: float = 1.0,
        use_channel_token_branch: bool = False,
        channel_token_layers: int = 3,
        channel_token_heads: int = 4,
        channel_token_alpha_init: float = 0.3,
    ):
        super().__init__()
        self.channel_names = list(channel_names)
        self.window = window
        self.d_model = d_model
        self.layer_dropout = float(layer_dropout)
        self.actuator_aug_rate = float(actuator_aug_rate)
        self.actuator_aug_max_channels = int(actuator_aug_max_channels)
        # default uniform layer weights; the v2 default is heavier on
        # downstream (state-bearing) layers because faults manifest there.
        if layer_score_weights is None:
            layer_score_weights = [1.0] * 5
        self.layer_score_weights = list(layer_score_weights)

        by_layer, channel_layer_id = channel_partition(channel_names)
        self.by_layer = by_layer
        self.channel_layer_id = channel_layer_id
        self.n_layers = len(by_layer)
        self.layer_sizes = [len(idxs) for idxs in by_layer]
        # Indices of "actuator" channels (command + servo + throttle) used by
        # the v2 training-time augmentation.
        self._actuator_idx = [
            i for i, name in enumerate(channel_names)
            if any(kw in name.lower()
                   for kw in ("command", "cmd_", "rc_", "servo", "throttle"))
        ]
        # cumulative order for the concatenated reconstruction
        ordered_channels: List[int] = []
        for idxs in by_layer:
            ordered_channels.extend(idxs)
        self.ordered_channels = ordered_channels
        # mapping from "ordered position" to original index for un-shuffling
        self.original_order = sorted(range(len(channel_names)),
                                     key=lambda c: ordered_channels.index(c))

        # v18: optionally REPLACE the iTransformer Linear(T->d_model) input
        # projection with a deep TimesNet-style encoder. The iT projection is
        # the "channel-as-token" inductive bias that v5+ uses; replacing it
        # with TN gives the encoder the same period-decomposition bias as the
        # TN baseline (which beats HALO on LOFO splits).
        self.use_tn_only_encoder = bool(use_tn_only_encoder)
        if self.use_tn_only_encoder:
            self.input_proj = None
            self.tn_encoder = TNBranch(
                n_channels=len(channel_names),
                d_model=d_model,
                window=window,
                n_layers=int(tn_encoder_layers),
                topk=int(tn_encoder_topk),
                hidden=int(tn_encoder_hidden),
                dropout=float(dropout),
            )
        else:
            # per-channel input projection: iTransformer style.
            self.input_proj = nn.Linear(window, d_model)
            self.tn_encoder = None
        # v5: optional multi-period 2D-conv embedding added to channel tokens.
        # Designed to give HALO TimesNet-style multi-scale temporal inductive
        # bias which Linear(T->d_model) lacks.
        self.use_multi_period = bool(use_multi_period)
        if self.use_multi_period:
            self.multi_period = MultiPeriodBlock(
                window=window,
                d_model=d_model,
                top_k=int(multi_period_topk),
                inner_dim=int(multi_period_inner),
                kernels=multi_period_kernels,
            )
        else:
            self.multi_period = None
        # v6: learnable gate scalar (sigmoid-bounded) so the model can
        # adaptively scale the multi-period contribution. Init logit so that
        # sigmoid(init) = `multi_period_gate_init` (default 0 -> contribution
        # disabled at init, model learns to use it).
        self.use_multi_period_gate = bool(multi_period_gate)
        if self.use_multi_period_gate:
            g0 = float(multi_period_gate_init)
            g0 = min(max(g0, 1e-4), 1.0 - 1e-4)
            logit = torch.log(torch.tensor(g0 / (1.0 - g0)))
            self.mp_gate_logit = nn.Parameter(logit)
        else:
            self.mp_gate_logit = None
        # v7: predictive head via input masking. Zero out the last
        # ``pred_horizon`` timesteps of the window before encoding. The recon
        # head still outputs the full window, so the loss against ``x_clean``
        # implicitly trains the model to FORECAST those steps from the past.
        # ``pred_score_weight`` up-weights the predicted (future) part during
        # anomaly_score so the predictive signal dominates the detector.
        self.pred_horizon = max(0, int(pred_horizon))
        self.pred_score_weight = float(pred_score_weight)
        # v9-B: TN-style multi-scale 2D conv stem on the (channel, time) map.
        # Adds cross-channel temporal-morphology bias on top of the
        # per-channel Linear and multi-period embeddings.
        self.use_tn_stem = bool(use_tn_stem)
        if self.use_tn_stem:
            self.tn_stem = TNStem(
                window=window,
                d_model=d_model,
                kernels=tn_stem_kernels,
                inner_per_branch=int(tn_stem_inner),
            )
        else:
            self.tn_stem = None
        # v9-B2: learnable scalar gate (sigmoid-bounded) on the TN stem.
        # ``tn_stem_gate_init`` is the *target sigmoid output* at init; we
        # store the corresponding logit. init=0.1 starts the stem at 10%
        # contribution so strat can shrink it toward 0 and elev can grow it.
        self.use_tn_stem_gate = bool(tn_stem_gate) and self.use_tn_stem
        if self.use_tn_stem_gate:
            g0 = float(tn_stem_gate_init)
            g0 = min(max(g0, 1e-4), 1.0 - 1e-4)
            logit = torch.log(torch.tensor(g0 / (1.0 - g0)))
            self.tn_stem_gate_logit = nn.Parameter(logit)
        else:
            self.tn_stem_gate_logit = None
        # v12: TN-style stacked SpectralBranch as a parallel encoder branch.
        # Gated sum into HALO's channel embedding so the model can dial it
        # up where useful (LOFO splits) and down where harmful (strat).
        self.use_tn_branch = bool(use_tn_branch)
        if self.use_tn_branch:
            self.tn_branch = TNBranch(
                n_channels=len(channel_names),
                d_model=d_model,
                window=window,
                n_layers=int(tn_branch_layers),
                topk=int(tn_branch_topk),
                hidden=int(tn_branch_hidden),
                dropout=float(dropout),
            )
        else:
            self.tn_branch = None
        self.use_tn_branch_gate = bool(tn_branch_gate) and self.use_tn_branch
        if self.use_tn_branch_gate:
            g0 = float(tn_branch_gate_init)
            g0 = min(max(g0, 1e-4), 1.0 - 1e-4)
            logit = torch.log(torch.tensor(g0 / (1.0 - g0)))
            self.tn_branch_gate_logit = nn.Parameter(logit)
        else:
            self.tn_branch_gate_logit = None
        # v13: per-sample data-dependent gate computed by a small MLP over
        # input summary statistics. Overrides scalar gate when enabled.
        self.use_tn_branch_data_gate = (
            bool(tn_branch_gate_data_dependent) and self.use_tn_branch
        )
        if self.use_tn_branch_data_gate:
            self.tn_branch_data_gate = DataDependentGate(
                n_channels=len(channel_names),
                hidden=int(tn_branch_gate_mlp_hidden),
                bias_init=float(tn_branch_gate_bias_init),
            )
        else:
            self.tn_branch_data_gate = None
        # v14: deterministic hard input-statistic gate (no learnable params).
        # Calibrated from train data after model construction.
        self.use_tn_branch_hard_gate = (
            bool(tn_branch_gate_hard_input) and self.use_tn_branch
        )
        if self.use_tn_branch_hard_gate:
            self.tn_branch_hard_gate = HardInputStatGate(
                n_channels=len(channel_names),
                min_active=float(tn_branch_gate_hard_min_active),
            )
        else:
            self.tn_branch_hard_gate = None
        # v15: epoch-warmup multiplier on the hard gate.
        # gate_effective = gate * clamp((epoch - start) / (full - start), 0, 1)
        # Set via set_current_epoch() from the trainer.
        self.tn_branch_gate_warmup_start = int(tn_branch_gate_warmup_start)
        self.tn_branch_gate_warmup_full = int(tn_branch_gate_warmup_full)
        self.register_buffer("_current_epoch", torch.tensor(0, dtype=torch.long))
        # v19: parallel LSTM forecast head — multi-view signal that complements
        # the recon-residual on engine-type LOFO failures (continuous decline).
        self.use_lstm_forecast = bool(use_lstm_forecast)
        if self.use_lstm_forecast:
            self.lstm_forecast = LSTMForecastHead(
                n_channels=len(channel_names),
                window=window,
                horizon=int(lstm_forecast_horizon),
                hidden=int(lstm_forecast_hidden),
                n_layers=int(lstm_forecast_layers),
                dropout=float(dropout),
            )
            self.lstm_score_weight = float(lstm_score_weight)
            self.lstm_loss_weight = float(lstm_loss_weight)
        else:
            self.lstm_forecast = None
            self.lstm_score_weight = 0.0
            self.lstm_loss_weight = 0.0
        # learnable per-layer position embedding (added to channel tokens)
        self.layer_embed = nn.Parameter(torch.zeros(self.n_layers, d_model))
        nn.init.normal_(self.layer_embed, std=0.02)

        self.encoders = nn.ModuleList([
            LayerEncoder(d_model, n_heads, n_encoder_blocks, dropout)
            for _ in range(self.n_layers)
        ])
        # Decoders only exist for L1..L_{N-1}; layer 0 has no upstream context
        self.decoders = nn.ModuleList([
            CrossLayerDecoder(d_model, n_heads, n_decoder_blocks, dropout)
            if L > 0 else nn.Identity()
            for L in range(self.n_layers)
        ])

        # per-layer reconstruction head: d_model -> window for each
        # channel in this layer.
        self.recon_heads = nn.ModuleList([
            nn.Linear(d_model, window) for _ in range(self.n_layers)
        ])

        # v26: parallel channel-as-token global self-attention branch + its
        # own reconstruction head. Lets ALL 16 channels attend to each other
        # in one block (iTransformer style), complementing HALO's 5-layer
        # hier-attn which only attends within-layer + cross-layer forward.
        # Output fused with hier recon via learnable sigmoid alpha.
        self.use_channel_token_branch = bool(use_channel_token_branch)
        if self.use_channel_token_branch:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(channel_token_heads),
                dim_feedforward=4 * d_model,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
            )
            self.channel_token_attn = nn.TransformerEncoder(
                enc_layer, num_layers=int(channel_token_layers)
            )
            self.channel_recon_head = nn.Linear(d_model, window)
            a0 = float(channel_token_alpha_init)
            a0 = min(max(a0, 1e-4), 1.0 - 1e-4)
            logit = torch.log(torch.tensor(a0 / (1.0 - a0)))
            self.channel_token_alpha_logit = nn.Parameter(logit)
        else:
            self.channel_token_attn = None
            self.channel_recon_head = None
            self.channel_token_alpha_logit = None

    # ------------------------------------------------------------------
    def _tokenize(self, x: torch.Tensor) -> List[torch.Tensor]:
        """``x`` shape ``(B, T, C)``. Returns a list of ``n_layers`` tensors,
        each shaped ``(B, n_layer_channels, d_model)`` with the layer
        embedding added."""
        B, T, C = x.shape
        # v18: TN encoder REPLACES iT Linear projection.
        if self.use_tn_only_encoder:
            proj = self.tn_encoder(x)                            # (B, C, d_model)
        else:
            # iTransformer-style channel-as-token projection.
            proj = self.input_proj(x.permute(0, 2, 1).contiguous())
        if self.multi_period is not None:
            mp = self.multi_period(x)
            if self.mp_gate_logit is not None:
                mp = torch.sigmoid(self.mp_gate_logit) * mp
            proj = proj + mp
        if self.tn_stem is not None:
            stem = self.tn_stem(x)
            if self.tn_stem_gate_logit is not None:
                stem = torch.sigmoid(self.tn_stem_gate_logit) * stem
            proj = proj + stem
        if self.tn_branch is not None:
            warmup_mul = 1.0
            if self.tn_branch_gate_warmup_full > self.tn_branch_gate_warmup_start:
                ep = float(self._current_epoch.item())
                w = (ep - self.tn_branch_gate_warmup_start) / float(
                    self.tn_branch_gate_warmup_full - self.tn_branch_gate_warmup_start)
                warmup_mul = max(0.0, min(1.0, w))
            # Skip the TN branch forward entirely when warmup is zero, so the
            # branch's dropout / FFT / convs do not consume RNG state and
            # perturb other layers' init/dropout streams (v15 lesson: even
            # forward with *0 multiplier shifted v5 ckpts' init).
            if warmup_mul > 0.0:
                br = self.tn_branch(x)                          # (B, C, d_model)
                if self.tn_branch_hard_gate is not None:
                    br = warmup_mul * self.tn_branch_hard_gate(x) * br
                elif self.tn_branch_data_gate is not None:
                    br = warmup_mul * self.tn_branch_data_gate(x) * br
                elif self.tn_branch_gate_logit is not None:
                    br = warmup_mul * torch.sigmoid(self.tn_branch_gate_logit) * br
                proj = proj + br
        out: List[torch.Tensor] = []
        for L, idxs in enumerate(self.by_layer):
            if not idxs:
                out.append(proj.new_zeros(B, 0, self.d_model))
                continue
            h = proj[:, idxs] + self.layer_embed[L].unsqueeze(0).unsqueeze(0)
            out.append(h)
        return out

    def _maybe_augment_actuators(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time augmentation: with probability ``actuator_aug_rate``,
        zero out 1..K randomly chosen actuator-layer channels in the input
        window (per sample, independently). Aligns the training distribution
        with the LOFO + actuator-mask deployment scenario.
        """
        if (not self.training) or self.actuator_aug_rate <= 0 or not self._actuator_idx:
            return x
        B = x.shape[0]
        # Per-sample independent decision.
        apply = torch.rand(B, device=x.device) < self.actuator_aug_rate
        if not apply.any():
            return x
        x_out = x.clone()
        K = max(1, min(self.actuator_aug_max_channels, len(self._actuator_idx)))
        for b in apply.nonzero(as_tuple=False).flatten().tolist():
            n_drop = int(torch.randint(low=1, high=K + 1, size=(1,)).item())
            chosen = torch.tensor(self._actuator_idx)[torch.randperm(len(self._actuator_idx))[:n_drop]]
            x_out[b, :, chosen] = 0.0
        return x_out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[-1] != len(self.channel_names) or x.shape[1] != self.window:
            raise ValueError(
                f"Expected (B, {self.window}, {len(self.channel_names)}); got {tuple(x.shape)}"
            )
        x = self._maybe_augment_actuators(x)
        # v7: mask the last pred_horizon steps to drive the recon head into
        # forecasting mode. Applied at both train and eval — anomalies in the
        # future window then surface as large prediction residual.
        if self.pred_horizon > 0:
            x_inp = x.clone()
            x_inp[:, -self.pred_horizon:, :] = 0.0
        else:
            x_inp = x
        B, T, _ = x_inp.shape
        per_layer_tokens = self._tokenize(x_inp)

        encoded: List[torch.Tensor] = []
        for L in range(self.n_layers):
            h = self.encoders[L](per_layer_tokens[L])
            encoded.append(h)

        recon_by_layer: Dict[int, torch.Tensor] = {}
        decoded: List[torch.Tensor] = []
        for L in range(self.n_layers):
            if L == 0 or all(t.shape[1] == 0 for t in encoded[:L]):
                d = encoded[L]
            else:
                # Stochastic layer dropout during training: drop one upstream layer
                upstream_list = [encoded[u] for u in range(L) if encoded[u].shape[1] > 0]
                if self.training and self.layer_dropout > 0 and len(upstream_list) > 1:
                    keep_mask = torch.rand(len(upstream_list)) > self.layer_dropout
                    if keep_mask.any():
                        upstream_list = [t for t, k in zip(upstream_list, keep_mask) if k]
                kv = torch.cat(upstream_list, dim=1) if upstream_list else encoded[L]
                d = self.decoders[L](encoded[L], kv)
            decoded.append(d)
            if encoded[L].shape[1] == 0:
                continue
            recon_layer = self.recon_heads[L](d)            # (B, n_l_ch, T)
            recon_layer = recon_layer.permute(0, 2, 1)      # (B, T, n_l_ch)
            recon_by_layer[L] = recon_layer

        # Assemble concatenated reconstruction in the *original* channel
        # order so downstream metrics that index by channel still work.
        # Match the per-layer recon dtype so fp16 training succeeds.
        any_recon = next(iter(recon_by_layer.values())) if recon_by_layer else x
        recon = torch.zeros(B, T, len(self.channel_names),
                            dtype=any_recon.dtype, device=any_recon.device)
        for L, idxs in enumerate(self.by_layer):
            if not idxs:
                continue
            recon[:, :, idxs] = recon_by_layer[L]

        # v26: channel-as-token global-attention branch. Operates on the same
        # iTransformer channel embedding (B, C, d_model), but allows ALL
        # channels to self-attend (no layer split) — captures pairwise
        # channel relationships that the hier path discards. Fuse via sigmoid
        # alpha so model can dial it up where useful (LOFO splits).
        if self.use_channel_token_branch and self.channel_token_attn is not None:
            if self.use_tn_only_encoder:
                proj_g = self.tn_encoder(x)                           # (B, C, d)
            else:
                proj_g = self.input_proj(x.permute(0, 2, 1).contiguous())
            z = self.channel_token_attn(proj_g)                       # (B, C, d)
            recon_ch = self.channel_recon_head(z)                     # (B, C, T)
            recon_ch = recon_ch.permute(0, 2, 1).contiguous()         # (B, T, C)
            alpha = torch.sigmoid(self.channel_token_alpha_logit)
            recon = (1.0 - alpha) * recon + alpha * recon_ch.to(recon.dtype)

        # dummy density head for trainer compatibility (no NLL used in HALO)
        mu = recon
        logvar = torch.zeros_like(recon)

        # v19: optional LSTM forecast head. Run on UNMASKED original input
        # so the forecast target is genuine future, not the pred_horizon mask.
        lstm_out = None
        if self.lstm_forecast is not None:
            lstm_out = self.lstm_forecast(x)

        return {
            "recon": recon,
            "recon_by_layer": recon_by_layer,
            "mu": mu,
            "logvar": logvar,
            "adj": None,
            "h": None,
            "lstm_forecast": lstm_out["forecast"] if lstm_out is not None else None,
            "lstm_target":   lstm_out["target"]   if lstm_out is not None else None,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(
        self,
        x: torch.Tensor,
        out: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        reduction: str = "channel_mean",
        topk: int = 3,
        spec_weight: float = 0.0,
        lstm_score_weight: float = 0.0,
    ) -> torch.Tensor:
        if out is None:
            out = self.forward(x)
        recon = out["recon"]
        residual = (x - recon).pow(2)
        # v10: spectral residual broadcast over the window as additive
        # per-(B, C) energy. The training loss already includes
        # freq_weight=0.3 so recon matches normal-flight spectra; on
        # anomalies the FFT magnitudes diverge — channel-wise mean
        # over frequency bins gives a scalar that catches stuck-actuator
        # and altered-period faults (rudder/elev/engine LOFO) that
        # squared time residual under-detects.
        if spec_weight > 0.0:
            x_fft = torch.fft.rfft(x, dim=1).abs()
            recon_fft = torch.fft.rfft(recon, dim=1).abs()
            spec_diff = (x_fft - recon_fft).pow(2).mean(dim=1, keepdim=True)
            residual = residual + spec_weight * spec_diff
        # v7: up-weight the masked (future) part of the window so the
        # forecasting residual carries more anomaly evidence than the
        # easier past-reconstruction residual.
        if self.pred_horizon > 0 and self.pred_score_weight > 0:
            H = self.pred_horizon
            w_future = 1.0 + self.pred_score_weight
            residual = residual.clone()
            residual[:, -H:, :] = residual[:, -H:, :] * w_future
        # v2: weighted per-layer reduction. The per-layer MSE is computed,
        # then summed weighted by ``self.layer_score_weights``. The total is
        # normalised so equal weights reproduce the v1 channel mean exactly.
        if reduction in ("channel_mean", "layer_weighted"):
            weights = torch.tensor(self.layer_score_weights, dtype=residual.dtype,
                                   device=residual.device)
            per_layer_means = []
            counts = []
            for L, idxs in enumerate(self.by_layer):
                if not idxs:
                    continue
                per_layer_means.append(residual[..., idxs].mean(dim=-1))
                counts.append(weights[L].item() * len(idxs))
            w = torch.tensor([self.layer_score_weights[L]
                              for L, idxs in enumerate(self.by_layer) if idxs],
                             dtype=residual.dtype, device=residual.device)
            stacked = torch.stack(per_layer_means, dim=-1)            # (B, T, n_used_layers)
            total = (stacked * w).sum(dim=-1) / max(w.sum().item(), 1e-6)
            score = alpha * total
        elif reduction == "channel_max":
            score = alpha * residual.amax(dim=-1)
        elif reduction == "channel_topk":
            # mean over top-k channels per timestep — designed for sparse-channel
            # faults (e.g. LOFO splits) where channel_mean dilutes the signal.
            k = max(1, min(int(topk), residual.shape[-1]))
            top_vals, _ = residual.topk(k, dim=-1)
            score = alpha * top_vals.mean(dim=-1)
        elif reduction == "layer_max":
            # per-layer mean then MAX over layers — picks the worst-recon layer
            # rather than averaging across layers (helps when a fault lives
            # entirely in one layer, e.g. command or state).
            per_layer_means = []
            for L, idxs in enumerate(self.by_layer):
                if not idxs:
                    continue
                per_layer_means.append(residual[..., idxs].mean(dim=-1))
            stacked = torch.stack(per_layer_means, dim=-1)
            score = alpha * stacked.amax(dim=-1)
        else:
            raise ValueError(reduction)
        # v21: LSTM forecast residual fusion. Per-window mean forecast MSE
        # added as a constant offset per window. Catches stuck-control faults
        # (rudder/elev LOFO) where the LSTM's "continue the temporal trend"
        # prior diverges from a frozen signal — recon residual under-detects
        # since HALO can still reproduce a flat output.
        if lstm_score_weight > 0.0 and self.lstm_forecast is not None:
            fc = out.get("lstm_forecast")
            tg = out.get("lstm_target")
            if fc is not None and tg is not None:
                fc_err = (fc - tg).pow(2).mean(dim=(1, 2))      # (B,)
                score = score + lstm_score_weight * fc_err.unsqueeze(-1)
        return score

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    ch = [
        "command/aileron", "command/elevator", "command/throttle", "command/rudder",
        "servo/aileron", "servo/elevator", "servo/throttle",
        "airspeed", "pressure/atm",
        "imu/gyro_x", "imu/gyro_y", "imu/gyro_z",
        "state/roll", "state/pitch", "state/yaw", "state/alt",
    ]
    T = 100
    m = HALO(channel_names=ch, window=T, d_model=64, n_heads=4,
             n_encoder_blocks=2, n_decoder_blocks=2)
    x = torch.randn(2, T, len(ch))
    out = m(x)
    print("recon", out["recon"].shape)
    print("recon_by_layer keys", list(out["recon_by_layer"].keys()))
    for L, r in out["recon_by_layer"].items():
        print(f"  L{L} recon {r.shape}")
    print("anomaly_score", m.anomaly_score(x, out).shape)
    print("params", f"{m.num_params() / 1e6:.2f}M")
