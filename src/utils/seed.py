"""Reproducibility helpers."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch (if available).

    Set ``deterministic=True`` only for final-result runs — it disables cuDNN
    autotuning and slows training noticeably.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.backends.cudnn.benchmark = True
    except ModuleNotFoundError:
        pass


def maybe_set_seed(seed: Optional[int], **kwargs) -> None:
    if seed is not None:
        set_seed(seed, **kwargs)
