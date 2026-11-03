"""Shared anomaly-score configuration helpers.

Training-time evaluation and post-hoc robustness evaluation must use the
same score composition.  The helper maps a stable YAML ``score`` block onto
the keyword names exposed by each model's ``anomaly_score`` method.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import numpy as np


def score_kwargs_from_config(
    cfg: Dict[str, Any], model: Optional[object] = None
) -> Dict[str, Any]:
    score_cfg = cfg.get("score", {}) or {}
    raw = {
        "alpha": float(score_cfg.get("rec_weight", score_cfg.get("alpha", 1.0))),
        "beta": float(score_cfg.get("density_weight", score_cfg.get("beta", 0.0))),
        "gamma": float(score_cfg.get("graph_weight", score_cfg.get("gamma", 0.0))),
        "delta_patch": float(
            score_cfg.get("patch_weight", score_cfg.get("delta_patch", 0.5))
        ),
        "delta_assoc": float(
            score_cfg.get("assoc_weight", score_cfg.get("delta_assoc", 0.5))
        ),
        "reduction": str(score_cfg.get("reduction", "channel_mean")),
        "spec_weight": float(score_cfg.get("spec_weight", 0.0)),
        "lstm_score_weight": float(score_cfg.get("lstm_score_weight", 0.0)),
    }
    if model is None:
        return raw

    params = inspect.signature(model.anomaly_score).parameters
    return {k: v for k, v in raw.items() if k in params}


def smooth_time_score(score: np.ndarray, window: int = 1) -> np.ndarray:
    """Causal trailing moving-average smoothing.

    The output at index ``t`` depends only on indices ``<= t``. Early samples
    use the available prefix rather than reflected or future padding.
    """
    window = int(window)
    if window <= 1:
        return np.asarray(score)
    values = np.asarray(score, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    out = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        out[index] = (prefix[index + 1] - prefix[start]) / (index + 1 - start)
    return out


def aggregate_window_scores(
    scores: np.ndarray,
    n_target: int,
    window: int,
    mode: str = "last",
) -> np.ndarray:
    """Project stride-1 window scores back to per-timestep scores."""
    n_w = scores.shape[0]
    full = np.zeros(n_target, dtype=np.float64)
    if n_w == 0:
        return full

    if mode == "last":
        last_step_scores = scores[:, -1]
        for w_idx in range(n_w):
            t_end = w_idx + window - 1
            if 0 <= t_end < n_target:
                full[t_end] = last_step_scores[w_idx]
        full[: window - 1] = full[window - 1]
        return full

    if mode not in {"mean", "max"}:
        raise ValueError(f"unknown eval.score_aggregation={mode!r}")

    if mode == "mean":
        denom = np.zeros(n_target, dtype=np.float64)
        for offset in range(window):
            end = min(offset + n_w, n_target)
            span = end - offset
            if span <= 0:
                continue
            full[offset:end] += scores[:span, offset]
            denom[offset:end] += 1.0
        mask = denom > 0
        full[mask] /= denom[mask]
        if not mask.all():
            full[~mask] = full[mask][0] if mask.any() else 0.0
        return full

    full.fill(-np.inf)
    for offset in range(window):
        end = min(offset + n_w, n_target)
        span = end - offset
        if span <= 0:
            continue
        full[offset:end] = np.maximum(full[offset:end], scores[:span, offset])
    finite = np.isfinite(full)
    if not finite.all():
        full[~finite] = full[finite][0] if finite.any() else 0.0
    return full


def project_window_scores(
    scores: np.ndarray,
    n_target: int,
    window: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    eval_cfg = cfg.get("eval", {}) or {}
    mode = str(eval_cfg.get("score_aggregation", "last"))
    smooth_window = int(eval_cfg.get("score_smooth_window", 1))
    if bool(eval_cfg.get("causal", True)) and mode != "last":
        raise ValueError(
            "causal evaluation requires eval.score_aggregation='last'; "
            f"received {mode!r}"
        )
    full = aggregate_window_scores(scores, n_target=n_target, window=window, mode=mode)
    return smooth_time_score(full, smooth_window)
