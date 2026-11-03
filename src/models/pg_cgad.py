"""PG-CGAD: Physics-Guided Channel Graph Anomaly Detector for UAV telemetry.

PG-CGAD addresses a UAV-specific gap left open by prior channel-graph
anomaly detectors: the learned inter-channel adjacency is fully
data-driven and does not exploit the well-defined UAV control
architecture ``command -> servo / control-surface -> aerodynamic ->
IMU -> state``. PG-CGAD injects this control-architecture prior into the
adjacency at three levels:

  (i)  a **hard prior mask** ``M_prior in {0,1}^{C x C}`` derived from the
       UAV channel taxonomy, which zeroes adjacency entries that violate
       the partial order;
  (ii) a **soft layer-distance prior** ``P_prior[i, j] = exp(-d_ij / tau)``
       that anchors short-range causal edges higher than long-range ones;
  (iii) a **prior-consistency regulariser** ``|| (A - P_prior) * M_prior ||_F^2``
       added to the training objective.

NOTEARS-style acyclicity is retained from the base channel-graph
architecture; the spectral branch is retained as a complementary
inductive bias.

Forward output keys (same shape contract as the base channel-graph model
so the existing scoring / evaluation utilities can be reused):

``recon``    : (B, T, C)   reconstructed window
``mu``       : (B, T, C)   per-timestep mean head
``logvar``   : (B, T, C)   per-timestep log-variance head
``adj``      : (C, C)      current channel adjacency (or None)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .modules.causal_attn import ChannelDAG, ChannelGraphAttention
from .modules.spectral import SpectralBranch


# ---------------------------------------------------------------------------
# UAV channel taxonomy
# ---------------------------------------------------------------------------


UAV_LAYERS: List[Tuple[int, Tuple[str, ...]]] = [
    (0, ("command", "setpoint", "cmd_", "rc_")),
    (1, ("servo", "surface", "throttle", "elev", "ail", "rud", "motor")),
    (2, ("airspeed", "pressure", "baro", "static")),
    (3, ("imu", "gyro", "accel", "mag", "ang_")),
    (4, ("state", "est_", "position", "pos_", "vel_",
         "roll", "pitch", "yaw", "alt", "lat", "lon")),
]


def assign_layers(channel_names: List[str]) -> List[int]:
    """Map every channel to a UAV control-architecture layer in [0, 4].

    Unknown channels default to the sink layer (4), letting them attend
    to everything; this is the safest default for unfamiliar data.
    """
    layers: List[int] = []
    for raw in channel_names:
        name = raw.lower()
        layer = 4
        for layer_id, kws in UAV_LAYERS:
            if any(kw in name for kw in kws):
                layer = layer_id
                break
        layers.append(layer)
    return layers


def build_hard_prior_mask(channel_names: List[str]) -> torch.Tensor:
    """``M_prior[i, j] = 1`` iff ``i`` is an allowed cause of ``j``
    (i.e. ``layer(i) <= layer(j)``). Diagonal is 1 (self-attention is
    always allowed). Returns a float tensor of shape ``(C, C)``.
    """
    layers = assign_layers(channel_names)
    C = len(layers)
    L = torch.tensor(layers, dtype=torch.long)
    mask = (L.unsqueeze(1) <= L.unsqueeze(0)).float()
    mask.fill_diagonal_(1.0)
    return mask


def build_soft_layer_prior(
    channel_names: List[str], tau: float = 1.0
) -> torch.Tensor:
    """Soft prior over allowed edges: ``P[i, j] = exp(-(L_j - L_i) / tau)``
    for ``L_i <= L_j``, zero otherwise. Diagonal is set to 0 because the
    soft prior describes *cross-channel* edges only; self-attention is
    handled separately by the hard mask.
    """
    layers = torch.tensor(assign_layers(channel_names), dtype=torch.float32)
    C = layers.shape[0]
    dist = layers.unsqueeze(0) - layers.unsqueeze(1)        # (C, C)
    allowed = (dist >= 0).float()
    soft = torch.exp(-dist.clamp(min=0.0) / max(tau, 1e-6)) * allowed
    soft.fill_diagonal_(0.0)
    return soft


# ---------------------------------------------------------------------------
# Physics-guided channel DAG
# ---------------------------------------------------------------------------


class PhysicsGuidedChannelDAG(ChannelDAG):
    """ChannelDAG extended with a soft prior-consistency penalty.

    Inherits the low-rank ``A = sigmoid(U V^T + b) * M_prior`` parameterisation
    from ``ChannelDAG`` and adds two buffers:

      - ``soft_prior`` ``(C, C)``: layer-distance soft prior over allowed edges.
      - ``allowed`` ``(C, C)``: float copy of the hard mask, for masking the
        consistency loss to the prior-admissible region only.

    The consistency loss is::

        L_phys = mean(((A - soft_prior) * allowed) ** 2)

    This is bounded in [0, 1] for any A and is invariant to the channel
    count, so the multiplier can be tuned on one dataset and transferred.
    """

    def __init__(
        self,
        n_channels: int,
        hard_prior_mask: torch.Tensor,
        soft_prior: torch.Tensor,
        rank: int = 16,
        init_logit_bias: float = -3.0,
        init_corr: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            n_channels=n_channels,
            rank=rank,
            prior_mask=hard_prior_mask,
            zero_diagonal=True,
            init_corr=init_corr,
            init_logit_bias=init_logit_bias,
        )
        self.register_buffer("soft_prior", soft_prior.float())
        self.register_buffer("allowed", hard_prior_mask.float())

    def prior_consistency_loss(self) -> torch.Tensor:
        adj = self.forward()
        diff = (adj - self.soft_prior) * self.allowed
        return diff.pow(2).mean()


# ---------------------------------------------------------------------------
# Encoder block (channel-as-token transformer)
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


class PGCGADLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = ChannelGraphAttention(d_model, n_heads, dropout=dropout)
        self.ffn = _FFN(d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), adj=adj)
        h = h + self.ffn(self.ln2(h))
        return h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class PGCGAD(nn.Module):
    """Physics-guided channel-graph anomaly detector.

    Parameters
    ----------
    channel_names : list[str]
        Used to build the hard prior mask and the soft layer-distance prior.
        If ``use_physics_prior`` is False the priors degenerate to all-ones /
        zeros and the model collapses to the data-driven channel-graph
        baseline used by prior work.
    """

    def __init__(
        self,
        channel_names: List[str],
        window: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        spectral_topk: int = 3,
        spectral_hidden: int = 32,
        use_dag: bool = True,
        use_spectral: bool = True,
        use_physics_prior: bool = True,
        soft_prior_tau: float = 1.0,
        dag_rank: int = 16,
        dag_init_logit_bias: float = -3.0,
    ):
        super().__init__()
        C = len(channel_names)
        self.channel_names = list(channel_names)
        self.n_channels = C
        self.window = window
        self.use_dag = use_dag
        self.use_spectral = use_spectral
        self.use_physics_prior = use_physics_prior

        self.input_proj = nn.Linear(window, d_model)
        self.channel_embed = nn.Parameter(torch.zeros(C, d_model))
        nn.init.normal_(self.channel_embed, std=0.02)

        if use_spectral:
            self.spectral = SpectralBranch(
                n_channels=C, d_model=d_model, window=window,
                topk=spectral_topk, hidden=spectral_hidden,
            )
        else:
            self.spectral = None

        if use_dag:
            if use_physics_prior:
                hard = build_hard_prior_mask(channel_names)
                soft = build_soft_layer_prior(channel_names, tau=soft_prior_tau)
                self.dag = PhysicsGuidedChannelDAG(
                    n_channels=C,
                    hard_prior_mask=hard,
                    soft_prior=soft,
                    rank=dag_rank,
                    init_logit_bias=dag_init_logit_bias,
                )
            else:
                self.dag = ChannelDAG(
                    n_channels=C, rank=dag_rank, prior_mask=None,
                    init_logit_bias=dag_init_logit_bias,
                )
        else:
            self.dag = None

        self.layers = nn.ModuleList(
            [PGCGADLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.recon_proj = nn.Linear(d_model, window)
        self.density_proj = nn.Linear(d_model, 2 * window)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[-1] != self.n_channels or x.shape[1] != self.window:
            raise ValueError(
                f"Expected (B, {self.window}, {self.n_channels}); got {tuple(x.shape)}"
            )
        h = self.input_proj(x.permute(0, 2, 1).contiguous())
        h = h + self.channel_embed.unsqueeze(0)
        if self.spectral is not None:
            h = h + self.spectral(x)

        adj = self.dag() if self.dag is not None else None
        for layer in self.layers:
            h = layer(h, adj=adj)
        h = self.norm(h)

        recon = self.recon_proj(h).permute(0, 2, 1).contiguous()
        density = self.density_proj(h)
        mu, logvar = density.chunk(2, dim=-1)
        mu = mu.permute(0, 2, 1).contiguous()
        logvar = logvar.permute(0, 2, 1).contiguous().clamp(-8.0, 8.0)
        return {"recon": recon, "mu": mu, "logvar": logvar, "adj": adj, "h": h}

    # ------------------------------------------------------------------
    # Loss hooks
    # ------------------------------------------------------------------

    def physics_prior_loss(self) -> Optional[torch.Tensor]:
        """Prior-consistency penalty hook for the trainer. Returns ``None``
        when no physics prior is configured (so the trainer can skip it
        without branching on the model class)."""
        if isinstance(self.dag, PhysicsGuidedChannelDAG):
            return self.dag.prior_consistency_loss()
        return None

    # ------------------------------------------------------------------
    # Anomaly scoring
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
    ) -> torch.Tensor:
        if out is None:
            out = self.forward(x)
        recon = out["recon"]
        residual = x - recon
        rec = residual.pow(2)
        if reduction == "channel_mean":
            rec_score = rec.mean(dim=-1)
        elif reduction == "channel_max":
            rec_score = rec.amax(dim=-1)
        else:
            raise ValueError(reduction)

        score = alpha * rec_score
        if beta > 0:
            mu, logvar = out["mu"], out["logvar"]
            inv_var = torch.exp(-logvar)
            nll = 0.5 * (logvar + (x - mu).pow(2) * inv_var)
            score = score + beta * nll.mean(dim=-1)

        adj = out.get("adj")
        if gamma > 0 and adj is not None:
            abs_e = residual.abs()
            graph = torch.einsum("btc,btd,cd->bt", abs_e, abs_e, adj)
            graph = graph / max(adj.shape[0], 1)
            score = score + gamma * graph
        return score

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_adjacency(self) -> Optional[torch.Tensor]:
        return self.dag() if self.dag is not None else None

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
    model = PGCGAD(channel_names=ch, window=T, d_model=64, n_heads=4, n_layers=2)
    x = torch.randn(2, T, len(ch))
    out = model(x)
    print("recon", out["recon"].shape)
    print("adj", out["adj"].shape, "density", out["adj"].mean().item())
    print("L_phys", model.physics_prior_loss().item())
    print("score", model.anomaly_score(x, out).shape)
    print("params", f"{model.num_params() / 1e6:.2f}M")

    # Verify hard mask zeros out illegal edges
    A = out["adj"].detach()
    M = build_hard_prior_mask(ch)
    n_off = int(((A > 1e-3) & (M < 0.5)).sum().item())
    print(f"illegal_active_edges={n_off} (should be 0)")
