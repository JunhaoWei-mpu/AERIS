"""Single-GPU training with validation-only checkpoint selection."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..losses.composite import (
    LossWeights,
    composite_loss,
    random_mask,
    synthetic_outliers,
)
from ..losses.dag_loss import DAGSchedule, PhysSchedule, dag_penalty
from ..utils.logging_utils import JsonLogger, get_logger
from ..utils.scoring import aggregate_window_scores, score_kwargs_from_config


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        eval_label: Optional[np.ndarray],
        out_dir: str | Path,
        cfg: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eval_label = eval_label
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.device = device

        train_cfg = cfg["train"]
        loss_cfg = cfg["loss"]

        self.epochs = int(train_cfg.get("epochs", 30))
        self.lr = float(train_cfg.get("lr", 3.0e-4))
        self.wd = float(train_cfg.get("weight_decay", 1.0e-4))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 0))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.precision = str(train_cfg.get("precision", "fp16"))
        self.use_amp = self.precision == "fp16" and str(device).startswith("cuda")
        self.log_every = int(train_cfg.get("log_every", 50))
        self.eval_every = max(1, int(train_cfg.get("eval_every", 1)))
        self.patience = max(1, int(train_cfg.get("patience", 5)))
        self.min_epochs = max(1, int(train_cfg.get("min_epochs", 8)))
        self.min_delta = float(train_cfg.get("min_delta", 0.0))
        if self.eval_label is not None and not bool(
            cfg.get("eval", {}).get("legacy_allow_test_selection", False)
        ):
            raise ValueError(
                "TAES training forbids label-aware checkpoint selection. "
                "Pass a nominal validation loader and eval_label=None."
            )

        self.weights = LossWeights(
            rec=float(loss_cfg.get("rec_weight", 1.0)),
            nll=float(loss_cfg.get("nll_weight", 0.0)),
            freq=float(loss_cfg.get("freq_weight", 0.3)),
            outlier_rate=float(loss_cfg.get("outlier_inject_rate", 0.10)),
            mask_rate=float(loss_cfg.get("mask_rate", 0.15)),
        )
        self.dag_sched = DAGSchedule(
            warmup_epochs=int(loss_cfg.get("dag_warmup_epochs", 10)),
            lambda_max=float(loss_cfg.get("dag_weight", 0.1)),
        )
        self.phys_sched = PhysSchedule(
            decay_epochs=int(loss_cfg.get("phys_decay_epochs", 0)),
            phys_w_max=float(loss_cfg.get("physics_weight", 0.0)),
            phys_w_min=float(loss_cfg.get("physics_weight_min", 0.0)),
        )

        self.optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
        if self.optimizer_name not in {"adam", "adamw"}:
            raise ValueError(
                f"unsupported optimizer {self.optimizer_name!r}; expected adam or adamw"
            )
        optimizer_class = (
            torch.optim.Adam
            if self.optimizer_name == "adam"
            else torch.optim.AdamW
        )
        self.optim = optimizer_class(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # Linear warmup + cosine decay
        total_steps = max(1, self.epochs * len(train_loader))

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step + 1) / float(max(1, self.warmup_steps))
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.log = get_logger("trainer")
        self.json = JsonLogger(self.out_dir / "log.jsonl")
        self.global_step = 0
        self.best_metric = np.inf
        self.best_epoch = -1
        self.score_kwargs = score_kwargs_from_config(cfg, self.model)
        self.score_aggregation = str(cfg.get("eval", {}).get("score_aggregation", "last"))

    # ------------------------------------------------------------------

    def _step(self, x: torch.Tensor, epoch: int) -> Dict[str, torch.Tensor]:
        x = x.to(self.device, non_blocking=True)
        # Augment
        x_input, mask_pos = random_mask(x, self.weights.mask_rate)
        x_input, outlier_mask = synthetic_outliers(
            x_input, self.weights.outlier_rate, self.weights.outlier_strength
        )
        # Combined emphasis mask (positions to up-weight)
        emphasis = (mask_pos | outlier_mask).any(dim=-1, keepdim=True).float()

        with autocast("cuda", enabled=self.use_amp):
            out = self.model(x_input)
            comp = composite_loss(
                model_out=out,
                x_clean=x,
                x_input=x_input,
                mask=emphasis,
                weights=self.weights,
            )
            total = comp["total"]
        # DAG penalty computed outside autocast — matrix_exp has no fp16 kernel
        if hasattr(self.model, "dag") and self.model.dag is not None:
            lam = self.dag_sched(epoch)
            if lam > 0 and out["adj"] is not None:
                h_sq = dag_penalty(out["adj"], mode="expm", square=True)
                total = total + lam * h_sq.to(total.dtype)
                comp["dag"] = h_sq.detach()
                comp["dag_lambda"] = torch.as_tensor(lam, device=total.device)
        # v19: LSTM forecast auxiliary loss. Model exposes lstm_loss_weight > 0
        # and lstm_forecast head when use_lstm_forecast=True.
        if hasattr(self.model, "lstm_forecast") and self.model.lstm_forecast is not None \
                and getattr(self.model, "lstm_loss_weight", 0.0) > 0:
            fc = out.get("lstm_forecast"); tg = out.get("lstm_target")
            if fc is not None and tg is not None:
                lstm_l = (fc - tg).pow(2).mean()
                total = total + float(self.model.lstm_loss_weight) * lstm_l.to(total.dtype)
                comp["lstm"] = lstm_l.detach()

        # PG-CGAD physics prior consistency penalty.
        # Model exposes physics_prior_loss() → Tensor or None. Trainer multiplies
        # by the (possibly epoch-decayed) phys_w so legacy CCG configs that
        # omit phys_decay_epochs fall through with a constant weight.
        phys_w = self.phys_sched(epoch)
        if phys_w > 0 and hasattr(self.model, "physics_prior_loss"):
            phys = self.model.physics_prior_loss()
            if phys is not None:
                total = total + phys_w * phys.to(total.dtype)
                comp["phys"] = phys.detach()
                comp["phys_lambda"] = torch.as_tensor(phys_w, device=total.device)

        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optim)
        self.scaler.update()
        self.scheduler.step()
        return comp

    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        metrics = None
        epochs_without_improvement = 0
        last_epoch = -1
        for epoch in range(self.epochs):
            last_epoch = epoch
            if hasattr(self.model, "_current_epoch"):
                self.model._current_epoch.fill_(epoch)
            self.model.train()
            t0 = time.time()
            running = {}
            for it, batch in enumerate(self.train_loader):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                comp = self._step(x, epoch)
                for k, v in comp.items():
                    if k.endswith("_w"):
                        continue
                    if isinstance(v, torch.Tensor):
                        running[k] = running.get(k, 0.0) + float(v.detach().item())
                self.global_step += 1
                if (it + 1) % self.log_every == 0:
                    avg = {k: running[k] / (it + 1) for k in running}
                    self.log.info(
                        f"epoch {epoch} it {it+1}/{len(self.train_loader)} "
                        f"lr {self.scheduler.get_last_lr()[0]:.2e} | "
                        + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    )
            avg = {k: running[k] / max(1, len(self.train_loader)) for k in running}
            avg["epoch"] = epoch
            avg["time"] = time.time() - t0
            self.json.log({"phase": "train", **avg})
            should_eval = (
                self.eval_loader is not None
                and ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1)
            )
            metrics = self.validate_nominal(epoch=epoch) if should_eval else None
            if metrics is not None:
                cur = float(metrics["val_nominal_loss"])
                if cur < self.best_metric - self.min_delta:
                    self.best_metric = cur
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "epoch": epoch,
                            "metrics": metrics,
                            "cfg": self.cfg,
                            "selection_source": "nominal_validation_loss",
                        },
                        self.out_dir / "best.pth",
                    )
                else:
                    epochs_without_improvement += 1
                if (
                    epoch + 1 >= self.min_epochs
                    and epochs_without_improvement >= self.patience
                ):
                    self.json.log(
                        {
                            "phase": "early_stop",
                            "epoch": epoch,
                            "best_epoch": self.best_epoch,
                            "best_val_nominal_loss": self.best_metric,
                        }
                    )
                    break
        if self.best_epoch < 0:
            raise RuntimeError(
                "no validation checkpoint was selected; a non-empty nominal "
                "validation loader is required"
            )
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": last_epoch,
                "metrics": metrics,
                "cfg": self.cfg,
                "selection_source": "nominal_validation_loss",
            },
            self.out_dir / "last.pth",
        )
        return {
            "best_val_nominal_loss": self.best_metric,
            "best_epoch": self.best_epoch,
            "last_epoch": last_epoch,
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate_nominal(self, epoch: int = -1) -> Optional[Dict[str, Any]]:
        """Evaluate a label-free nominal validation objective."""
        if self.eval_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        for batch in self.eval_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.use_amp):
                out = self.model(x)
                per_sample = (out["recon"] - x).pow(2).mean(dim=(1, 2))
                if (
                    out.get("lstm_forecast") is not None
                    and out.get("lstm_target") is not None
                ):
                    forecast = (
                        out["lstm_forecast"] - out["lstm_target"]
                    ).pow(2).mean(dim=(1, 2))
                    per_sample = per_sample + float(
                        getattr(self.model, "lstm_loss_weight", 0.0)
                    ) * forecast
            total_loss += float(per_sample.sum().item())
            n_samples += int(per_sample.numel())
        flat = {
            "epoch": epoch,
            "val_nominal_loss": total_loss / max(1, n_samples),
            "n_val_windows": n_samples,
        }
        self.json.log({"phase": "eval", **flat})
        self.log.info(
            "VAL ep "
            + " ".join(f"{k}={v:.4f}" for k, v in flat.items() if isinstance(v, float))
        )
        (self.out_dir / f"val_ep{epoch}.json").write_text(
            json.dumps(flat, indent=2)
        )
        return flat

    def _aggregate_window_scores(
        self, scores: np.ndarray, n_target: int, window: int
    ) -> np.ndarray:
        return aggregate_window_scores(
            scores, n_target=n_target, window=window, mode=self.score_aggregation
        )
