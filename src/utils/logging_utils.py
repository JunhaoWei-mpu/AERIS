"""Logging that always reports to a JSON sidecar in addition to stdout."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def get_logger(name: str = "ccg_msd", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(h)
    log.propagate = False
    return log


class JsonLogger:
    """Append one JSON record per line to a sidecar file. Cheap, grep-friendly."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        record = {"ts": time.time(), **payload}
        if step is not None:
            record["step"] = step
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
