"""ALFA dataset loader and stratified split protocol.

Raw ALFA (CMU AirLab) ships as 47 flights in ROS-bag / CSV / .mat formats,
with eight fault categories (full power loss; stuck-at-zero on elevator,
aileron, rudder; combined; and a normal-flight reference set). Each fault
flight has a metadata-recorded *injection timestamp*; samples after that
timestamp are anomalous, samples before are nominal.

This module defines the **processed format** we commit to using throughout
the project. All training and evaluation reads from this format; the raw
ROS-bag conversion is handled separately in
``src/data/preprocess_alfa.py`` (to be added).

Processed layout
----------------
``data/processed/alfa/`` contains one ``.npz`` per flight:

    {flight_id}.npz
        x            : (T, D) float32     z-scored sensor readings
        label        : (T,)   int8        0 = nominal, 1 = post-injection
        t            : (T,)   float32     time (seconds since flight start)
        fault_type   : str    one of {nominal, engine, elev, aileron, rudder, mixed, ...}
        channel_names: list[str]
        sample_rate  : float  Hz  (resampled to a fixed rate during preprocess)

Plus a top-level ``meta.json`` with the global channel list (so feature
ordering is consistent across flights) and the resampling rate.

Stratified split
----------------
We propose ``ALFA-strat``: per fault category, flights are randomly
partitioned 60/20/20 into train / val / test (seed = 0). The split is
released as ``splits/alfa_strat.json`` and treated as a benchmark artefact.
We also support ``ALFA-lofo`` (leave-one-fault-out): for each fault
category, that category is held out at test time. This stresses
generalisation to unseen fault classes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class ALFAFlight:
    flight_id: str
    fault_type: str
    x: np.ndarray            # (T, D)
    label: np.ndarray        # (T,)
    t: np.ndarray            # (T,)
    sample_rate: float
    channel_names: List[str]

    @property
    def is_nominal(self) -> bool:
        return self.fault_type == "nominal"

    @property
    def n_timesteps(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])


def load_flight(npz_path: Path) -> ALFAFlight:
    npz_path = Path(npz_path)
    raw = np.load(npz_path, allow_pickle=True)
    return ALFAFlight(
        flight_id=npz_path.stem,
        fault_type=str(raw["fault_type"].item()),
        x=raw["x"].astype(np.float32),
        label=raw["label"].astype(np.int8),
        t=raw["t"].astype(np.float32),
        sample_rate=float(raw["sample_rate"].item()),
        channel_names=list(map(str, raw["channel_names"].tolist())),
    )


def load_all(processed_dir: str | Path) -> List[ALFAFlight]:
    processed_dir = Path(processed_dir)
    return [load_flight(p) for p in sorted(processed_dir.glob("*.npz"))]


# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------


def _stratified_partition(
    items: List[str], ratios: Tuple[float, float, float], rng: np.random.Generator
) -> Tuple[List[str], List[str], List[str]]:
    items = sorted(items)  # deterministic ordering before shuffle
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return [], [], []
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(0, int(round(ratios[1] * n)))
    # ensure at least 1 in test if we have ≥3
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_train -= 1
        n_test = n - n_train - n_val
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def make_strat_split(
    flights: List[ALFAFlight],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Stratified-by-fault-type 60/20/20 split.

    Returns a dict with keys ``train``, ``val``, ``test``, each a list of
    flight_ids. Within each fault category, flights are randomly partitioned
    in fixed ratios.
    """
    by_fault: Dict[str, List[str]] = defaultdict(list)
    for fl in flights:
        by_fault[fl.fault_type].append(fl.flight_id)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for fault, ids in sorted(by_fault.items()):
        tr, va, te = _stratified_partition(ids, ratios, rng)
        train.extend(tr)
        val.extend(va)
        test.extend(te)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test), "protocol": "alfa_strat", "seed": seed, "ratios": list(ratios)}


def make_lofo_split(flights: List[ALFAFlight], held_out_fault: str) -> Dict[str, List[str]]:
    """Leave-one-fault-out split: ``held_out_fault`` flights → test;
    other faults' flights → train; nominal flights → 70/30 train/val for
    stability in DAG learning."""
    train, val, test = [], [], []
    rng = np.random.default_rng(0)
    nominal = sorted(fl.flight_id for fl in flights if fl.fault_type == "nominal")
    rng.shuffle(nominal)
    cut = max(1, int(round(0.7 * len(nominal))))
    train.extend(nominal[:cut])
    val.extend(nominal[cut:])
    for fl in flights:
        if fl.fault_type == "nominal":
            continue
        if fl.fault_type == held_out_fault:
            test.append(fl.flight_id)
        else:
            train.append(fl.flight_id)
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "protocol": "alfa_lofo",
        "held_out_fault": held_out_fault,
    }


def save_split(split: Dict[str, List[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(split, f, indent=2, sort_keys=True)


def load_split(path: str | Path) -> Dict[str, List[str]]:
    path = Path(path)
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Building train / test arrays from a split
# ---------------------------------------------------------------------------


def concat_flights(
    flights: List[ALFAFlight], ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a list of flights into ``(x, label, segment_offsets)``.

    ``segment_offsets`` is a 1-D int array marking the start index of each
    flight in the concatenated tensor — needed downstream so that sliding
    windows do not cross flight boundaries.
    """
    by_id = {fl.flight_id: fl for fl in flights}
    arrs, labs, offs = [], [], [0]
    for fid in ids:
        if fid not in by_id:
            continue
        fl = by_id[fid]
        arrs.append(fl.x)
        labs.append(fl.label)
        offs.append(offs[-1] + fl.n_timesteps)
    if not arrs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.array(offs, dtype=np.int64),
        )
    x = np.concatenate(arrs, axis=0).astype(np.float32)
    y = np.concatenate(labs, axis=0).astype(np.int8)
    return x, y, np.array(offs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Channel-causality prior mask (UAV physics)
# ---------------------------------------------------------------------------


def build_uav_prior_mask(channel_names: List[str]) -> np.ndarray:
    """Construct a soft (0/1) prior mask over channels encoding the UAV
    causal chain: command → control_surface → aero_response → IMU → state.

    Mask convention: ``M[i, j] = 1`` allows channel ``j`` to attend to
    channel ``i`` (i.e. ``i`` is allowed to be a *cause* of ``j``).

    Layers (in causal order):
        L0  command_*, setpoint_*
        L1  servo_*, surface_*, throttle_*
        L2  airspeed_*, pressure_*, baro_*
        L3  imu_*, gyro_*, accel_*, mag_*
        L4  state_*, est_*, position_*, vel_*, roll, pitch, yaw, alt

    A channel may attend to its own layer and earlier layers. Unknown
    channels default to layer 4 (sink) so they can attend to everything.
    """
    LAYERS: List[Tuple[int, Tuple[str, ...]]] = [
        (0, ("command", "setpoint", "cmd_", "rc_")),
        (1, ("servo", "surface", "throttle", "elev", "ail", "rud", "motor")),
        (2, ("airspeed", "pressure", "baro", "static")),
        (3, ("imu", "gyro", "accel", "mag", "ang_")),
        (4, ("state", "est_", "position", "pos_", "vel_", "roll", "pitch", "yaw", "alt", "lat", "lon")),
    ]

    def layer_of(name: str) -> int:
        n = name.lower()
        for layer_id, kws in LAYERS:
            if any(kw in n for kw in kws):
                return layer_id
        return 4

    layers = np.array([layer_of(c) for c in channel_names])
    C = len(channel_names)
    M = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        for j in range(C):
            if layers[i] <= layers[j]:
                M[i, j] = 1.0
    np.fill_diagonal(M, 1.0)  # always allow self-attention
    return M


# ---------------------------------------------------------------------------
# Synthetic generator (for pipeline tests when raw ALFA not yet processed)
# ---------------------------------------------------------------------------


def synth_flight(
    flight_id: str,
    fault_type: str,
    n_timesteps: int = 600,
    sample_rate: float = 50.0,
    seed: int = 0,
) -> ALFAFlight:
    """Generate a plausible-shape fake flight for unit tests / smoketests."""
    rng = np.random.default_rng(seed)
    channels = [
        "command/throttle",
        "command/elev",
        "servo/elev",
        "servo/aileron",
        "airspeed",
        "pressure/static",
        "imu/gyro_x",
        "imu/gyro_y",
        "imu/gyro_z",
        "imu/accel_x",
        "imu/accel_y",
        "imu/accel_z",
        "state/roll",
        "state/pitch",
        "state/yaw",
        "state/alt",
    ]
    D = len(channels)
    t = np.arange(n_timesteps, dtype=np.float32) / sample_rate
    base = 0.3 * rng.standard_normal((n_timesteps, D)).astype(np.float32)
    # Slow oscillation in command → propagated with delay to surface, IMU, state.
    cmd = 0.5 * np.sin(2 * np.pi * 0.3 * t).astype(np.float32)
    base[:, 0] += cmd
    base[:, 1] += np.roll(cmd, 1)
    base[:, 2] += np.roll(cmd, 3)
    base[:, 6] += np.roll(cmd, 5)
    base[:, 12] += np.roll(cmd, 8)
    label = np.zeros(n_timesteps, dtype=np.int8)
    if fault_type != "nominal":
        # inject fault at 60% point; channels 1..3 stuck-at-zero for surface fault
        inj = int(0.6 * n_timesteps)
        label[inj:] = 1
        if fault_type == "elev":
            base[inj:, 1] = 0.0
            base[inj:, 2] = 0.0
        elif fault_type == "engine":
            base[inj:, 0] = 0.0
        # IMU/state still respond → that "broken" propagation is the signal
    return ALFAFlight(
        flight_id=flight_id,
        fault_type=fault_type,
        x=base,
        label=label,
        t=t,
        sample_rate=sample_rate,
        channel_names=channels,
    )


def synth_dataset(n_per_fault: int = 6) -> List[ALFAFlight]:
    out = []
    for ft in ("nominal", "engine", "elev", "aileron", "rudder"):
        for k in range(n_per_fault):
            out.append(synth_flight(f"{ft}_{k:02d}", ft, seed=hash((ft, k)) % (2**31)))
    return out
