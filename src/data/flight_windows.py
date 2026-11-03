"""Flight-contiguous, leakage-safe windows for TAES UAV experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .alfa import ALFAFlight


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "count": self.count.astype(int).tolist(),
        }


@dataclass(frozen=True)
class WindowRef:
    flight_index: int
    start: int
    end: int


def fit_nominal_standardizer(
    flights: Sequence[ALFAFlight], flight_ids: Iterable[str]
) -> Standardizer:
    """Fit channel statistics from finite, nominal training samples only."""
    selected = set(flight_ids)
    if not selected:
        raise ValueError("cannot fit standardizer without training flight IDs")
    n_channels = flights[0].n_features
    count = np.zeros(n_channels, dtype=np.int64)
    total = np.zeros(n_channels, dtype=np.float64)
    total_sq = np.zeros(n_channels, dtype=np.float64)
    used = 0
    for flight in flights:
        if flight.flight_id not in selected:
            continue
        mask = flight.label == 0
        values = np.asarray(flight.x[mask], dtype=np.float64)
        if values.size == 0:
            continue
        finite = np.isfinite(values)
        count += finite.sum(axis=0)
        total += np.where(finite, values, 0.0).sum(axis=0)
        total_sq += np.where(finite, values * values, 0.0).sum(axis=0)
        used += int(mask.sum())
    if used == 0 or np.any(count == 0):
        raise RuntimeError(
            "training split contains no finite nominal samples for every channel"
        )
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-8, 1.0, std)
    return Standardizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=count,
    )


def contiguous_segments(flight: ALFAFlight) -> list[tuple[int, int]]:
    """Return half-open segments with finite, monotonic, near-regular time."""
    n = flight.n_timesteps
    if n == 0:
        return []
    t = np.asarray(flight.t, dtype=np.float64)
    if t.shape != (n,):
        raise ValueError(f"{flight.flight_id}: invalid timestamp shape {t.shape}")
    expected = 1.0 / float(flight.sample_rate)
    delta = np.diff(t)
    good = (
        np.isfinite(delta)
        & (delta > 0.0)
        & (delta <= max(1.5 * expected, expected + 1e-6))
    )
    breaks = np.flatnonzero(~good) + 1
    bounds = np.concatenate(([0], breaks, [n]))
    return [
        (int(start), int(end))
        for start, end in zip(bounds[:-1], bounds[1:])
        if end > start
    ]


class FlightWindowDataset(Dataset):
    """Sliding windows that never cross flight or timestamp boundaries.

    With ``nominal_only=True``, a window is included only if every label in
    that window is zero. Removing an anomalous interval therefore cannot
    concatenate the samples on either side of the interval.
    """

    def __init__(
        self,
        flights: Sequence[ALFAFlight],
        flight_ids: Iterable[str],
        standardizer: Standardizer,
        window: int,
        stride: int,
        nominal_only: bool,
    ):
        self.window = int(window)
        self.stride = int(stride)
        if self.window <= 0 or self.stride <= 0:
            raise ValueError("window and stride must be positive")
        selected = set(flight_ids)
        self.flights = [
            flight for flight in flights if flight.flight_id in selected
        ]
        missing = selected - {flight.flight_id for flight in self.flights}
        if missing:
            raise KeyError(f"split refers to missing flights: {sorted(missing)[:5]}")
        self.standardizer = standardizer
        self.nominal_only = bool(nominal_only)
        self.refs: list[WindowRef] = []
        for flight_index, flight in enumerate(self.flights):
            if flight.n_features != standardizer.mean.size:
                raise ValueError(
                    f"{flight.flight_id}: channel count does not match standardizer"
                )
            anomaly_prefix = np.concatenate(
                ([0], np.cumsum(flight.label != 0, dtype=np.int64))
            )
            for seg_start, seg_end in contiguous_segments(flight):
                last_start = seg_end - self.window
                if last_start < seg_start:
                    continue
                for start in range(seg_start, last_start + 1, self.stride):
                    end = start + self.window
                    if self.nominal_only and (
                        anomaly_prefix[end] - anomaly_prefix[start]
                    ):
                        continue
                    self.refs.append(WindowRef(flight_index, start, end))
        if not self.refs:
            raise RuntimeError("window construction produced an empty dataset")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> torch.Tensor:
        ref = self.refs[index]
        flight = self.flights[ref.flight_index]
        values = self.standardizer.transform(flight.x[ref.start : ref.end])
        return torch.from_numpy(values)

    @property
    def channel_names(self) -> list[str]:
        return list(self.flights[0].channel_names)

    def flight_window_counts(self) -> dict[str, int]:
        counts = {flight.flight_id: 0 for flight in self.flights}
        for ref in self.refs:
            counts[self.flights[ref.flight_index].flight_id] += 1
        return counts


def project_last_window_scores(
    dataset: FlightWindowDataset, window_scores: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Project each window's last score to its causal flight-end timestamp."""
    values = np.asarray(window_scores)
    if values.ndim == 2:
        values = values[:, -1]
    if values.ndim != 1 or values.size != len(dataset):
        raise ValueError(
            f"expected {len(dataset)} window scores, received {values.shape}"
        )
    traces: dict[str, dict[str, np.ndarray]] = {}
    for flight in dataset.flights:
        traces[flight.flight_id] = {
            "score": np.full(flight.n_timesteps, np.nan, dtype=np.float64),
            "label": flight.label.astype(np.int8, copy=True),
            "time": flight.t.astype(np.float64, copy=True),
            "valid": np.zeros(flight.n_timesteps, dtype=bool),
        }
    for ref, score in zip(dataset.refs, values):
        flight = dataset.flights[ref.flight_index]
        timestep = ref.end - 1
        trace = traces[flight.flight_id]
        trace["score"][timestep] = float(score)
        trace["valid"][timestep] = np.isfinite(score)
    return traces


def concatenate_valid_traces(
    traces: dict[str, dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for flight_id in sorted(traces):
        trace = traces[flight_id]
        valid = np.asarray(trace["valid"], dtype=bool)
        labels.append(np.asarray(trace["label"])[valid])
        scores.append(np.asarray(trace["score"])[valid])
    if not labels:
        return np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float64)
    return np.concatenate(labels), np.concatenate(scores)
