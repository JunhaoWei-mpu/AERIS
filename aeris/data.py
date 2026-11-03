"""Prepare public CSV exports using the existing causal telemetry adapter."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from src.data.tvt_raw import sha256
from .protocol import ROOT


def guard_alfa(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=False)
    meta = json.loads((source / "meta.json").read_text())
    parent = sha256(source / "meta.json")
    for record in meta["flights"]:
        old = source / (record["flight_id"] + ".npz")
        if sha256(old) != record["npz_sha256"]:
            raise ValueError("Changed raw-rebuilt file")
        with np.load(old, allow_pickle=False) as d:
            arrays = {k: d[k].copy() for k in d.files}
        onset = record["onset_relative_s"]
        uncertain = np.zeros(len(arrays["t"]), bool) if onset is None else (
            (arrays["t"] >= onset - .2) & (arrays["t"] < onset))
        if np.any(uncertain & (arrays["label"] != 0)):
            raise ValueError("Uncertainty interval contains a native fault label")
        arrays["native_label"] = arrays["label"].copy()
        arrays["label_uncertain"] = uncertain
        arrays["label"][uncertain] = -1
        arrays["nominal_eligible"][uncertain] = False
        output = target / old.name
        np.savez_compressed(output, **arrays)
        record.update(npz_sha256=sha256(output),
                      nominal_eligible_samples=int(arrays["nominal_eligible"].sum()),
                      uncertain_samples=int(uncertain.sum()))
    meta.update(schema="tvt_v2_onset_guard", parent_meta_sha256=parent,
                onset_guard_s=.2, onset_guard_source="https://arxiv.org/pdf/1907.06268")
    (target / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def prepare(dataset, raw, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing processed-data directory")
    intermediate = output.with_name(output.name + "_before_onset_guard") if dataset == "alfa" else output
    if intermediate.exists():
        raise FileExistsError("Intermediate raw-rebuild directory already exists")
    subprocess.run([sys.executable, str(ROOT / "scripts/rebuild_tvt_data.py"),
                    "--dataset", dataset, "--raw", str(Path(raw).resolve()),
                    "--out", str(intermediate), "--max-age", "1.0"], cwd=ROOT, check=True)
    if dataset == "alfa":
        guard_alfa(intermediate, output)
    print(output)
