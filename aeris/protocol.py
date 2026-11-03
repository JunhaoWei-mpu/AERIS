"""Portable data membership for the frozen, source-separated protocol."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from src.data.tvt_raw import sha256
from src.data.tvt_dataset import load_tvt_flight, fit_tvt_standardizer, TVTWindowDataset
from src.models.tvt_control_chain import SEMANTICS
from scripts.rebuild_tvt_data import ALFA_CHANNELS, RFLY_CHANNELS

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("selection", "regression", "tail", "development")
SPLITS = {
    "alfa": ("strat", "lofo_engine", "lofo_aileron", "lofo_elev", "lofo_rudder"),
    "rflymad": ("strat", "lofo_motor", "lofo_gyro", "lofo_baro", "lofo_accel"),
}


def read_split(dataset, setting):
    if setting not in SPLITS[dataset]:
        raise ValueError(f"Unknown {dataset} setting: {setting}")
    path = ROOT / "splits" / f"{dataset}_{setting}.json"
    split = json.loads(path.read_text())
    groups = [set(split["source_groups"][p]) for p in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Source recordings overlap across partitions")
    return split, path


def prepare_nominal(root, dataset, setting):
    """Open training/nominal-validation files, never the test files."""
    root = Path(root)
    split, split_path = read_split(dataset, setting)
    meta = {f["flight_id"]: f for f in json.loads((root / "meta.json").read_text())["flights"]}
    expected = [c[0] for c in ALFA_CHANNELS] if dataset == "alfa" else RFLY_CHANNELS
    flights, hashes = [], {}
    for name in sorted(set(split["train"] + split["val"])):
        path = root / (name + ".npz")
        hashes[name] = sha256(path)
        if hashes[name] != meta[name]["npz_sha256"]:
            raise ValueError(f"Data checksum mismatch: {name}")
        flight = load_tvt_flight(path)
        if flight.channel_names != expected or flight.sample_rate != SEMANTICS[dataset].sample_rate:
            raise ValueError(f"Channel schema/sample rate mismatch: {name}")
        flights.append(flight)
    scaler = fit_tvt_standardizer(flights, split["train"])
    train = TVTWindowDataset(flights, split["train"], scaler, 100, 5, True)
    times, grouped = {}, {}
    for flight in flights:
        if flight.flight_id not in split["val"]:
            continue
        t = int(meta[flight.flight_id]["start_ns"]) + np.rint(flight.t * 1e9).astype(np.int64)
        times[flight.flight_id] = t
        eligible = flight.nominal_eligible & np.isfinite(flight.x).all(1)
        grouped.setdefault(split["flight_to_group"][flight.flight_id], []).append(t[eligible])
    boundaries = {}
    for group, arrays in grouped.items():
        t = np.unique(np.concatenate(arrays))
        if not len(t) or (dataset == "alfa" and len(t) < 800):
            raise ValueError(f"Insufficient nominal calibration support: {group}")
        boundaries[group] = [int(t[int(len(t) * q)]) for q in (.25, .5, .75)]
    parts = [[] for _ in PHASES]
    for flight in flights:
        if flight.flight_id not in times:
            continue
        t = times[flight.flight_id]
        b = boundaries[split["flight_to_group"][flight.flight_id]]
        role = np.searchsorted(b, t, side="right")
        allowed = flight.nominal_eligible.copy()
        for boundary in b:
            allowed &= np.abs(t - boundary) > 2_000_000_000
        for i in range(4):
            parts[i].append(replace(flight, nominal_eligible=allowed & (role == i)))
    phases = {key: TVTWindowDataset(fs, split["val"], scaler, 100, 1, True)
              for key, fs in zip(PHASES, parts)}
    values = [f.x[f.nominal_eligible & np.isfinite(f.x).all(1)]
              for f in flights if f.flight_id in split["train"]]
    lo = np.min(np.stack([v.min(0) for v in values if len(v)]), 0)
    hi = np.max(np.stack([v.max(0) for v in values if len(v)]), 0)
    constant = np.flatnonzero(lo == hi).tolist()
    lineage = dict(data_hashes=hashes, split_sha256=sha256(split_path),
                   calibration_boundaries_ns=boundaries, test_files_opened=False)
    return split, scaler, train, phases, meta, constant, lineage


def common_window_support(ds):
    """Inherited 199-sample comparison support; actual model input is 100."""
    valid = np.zeros(len(ds), bool)
    last, count = {}, {}
    for i, ref in enumerate(ds.refs):
        f = ref.flight_index
        count[f] = count.get(f, 0) + 1 if last.get(f) == ref.end - 1 else 1
        last[f] = ref.end
        valid[i] = count[f] >= 100
    return valid


def development_roles(ds, split, meta, dataset):
    valid = common_window_support(ds)
    groups = np.array([split["flight_to_group"][ds.flights[r.flight_index].flight_id]
                       for r in ds.refs])
    times = np.array([int(meta[ds.flights[r.flight_index].flight_id]["start_ns"])
                      + round(ds.flights[r.flight_index].t[r.end - 1] * 1e9)
                      for r in ds.refs], dtype=np.int64)
    starts = times - 198 * round(1e9 / SEMANTICS[dataset].sample_rate)
    role = np.full(len(ds), -1, np.int8)
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        cut = int(np.median(times[ix]))
        role[ix[valid[ix] & (times[ix] < cut - 1_000_000_000)]] = 0
        role[ix[valid[ix] & (starts[ix] > cut + 1_000_000_000)]] = 1
    return role


def nominal_library(train, scaler, split, meta):
    ds = TVTWindowDataset(train.flights, split["train"], scaler, 100, 100, True)
    records = []
    for i, ref in enumerate(ds.refs):
        flight = ds.flights[ref.flight_index]
        origin = int(meta[flight.flight_id]["start_ns"])
        records.append((split["flight_to_group"][flight.flight_id],
                        origin + round(flight.t[ref.start] * 1e9),
                        origin + round(flight.t[ref.end - 1] * 1e9), i))
    selected, last = [], {}
    for group, start, end, i in sorted(records):
        if start > last.get(group, -2**63):
            selected.append(i)
            last[group] = end
    ds.refs = [ds.refs[i] for i in selected]
    return ds
