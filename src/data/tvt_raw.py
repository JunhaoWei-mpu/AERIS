"""Raw-source auditing and causal sampling for the TVT recovery.

No normalization, fitted statistics, labels, or future values enter resampling.
Legacy channel names are deliberately NOT used as assertions about hardware.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def causal_hold(time, values, grid, *, max_age):
    """Last received sample, with explicit missing/stale evidence.

    Times must share a clock and units. Duplicate timestamps use their last
    recorded value; an invalid most-recent reading is not back/forward filled.
    Returned ages are independent of signal values and fault annotations.
    """
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if time.ndim != 1 or values.shape[0] != time.size or grid.ndim != 1:
        raise ValueError('incompatible sample shapes')
    if not time.size or not np.isfinite(time).all() or np.any(np.diff(time) < 0):
        raise ValueError('sample timestamps must be finite, nonempty, sorted')
    if not np.isfinite(grid).all() or np.any(np.diff(grid) < 0) or max_age <= 0:
        raise ValueError('invalid grid or max_age')
    index = np.searchsorted(time, grid, side='right') - 1
    safe_index = np.maximum(index, 0)
    age = grid - time[safe_index]
    valid = (index >= 0) & (age <= max_age)
    result = values[safe_index].copy()
    result[~valid] = np.nan
    age[index < 0] = np.inf
    return result.astype(np.float32), age


def read_topic(path: Path, clock: str):
    d = pd.read_csv(path, low_memory=False)
    if clock not in d or d.empty:
        raise ValueError(f'missing clock in {path}')
    if not pd.to_numeric(d[clock], errors='coerce').notna().all():
        raise ValueError(f'invalid clock in {path}')
    return d.sort_values(clock, kind='stable').reset_index(drop=True)


def alfa_fault_annotation(folder: Path):
    """Official processed ALFA publishes fault topics only during faults.

    Codes 1/2/3 identify fault modes, not a Boolean detection target. Use the
    first positive fault-status message, retaining ALL affected families.
    Unknown or missing ground truth is ineligible, never a midpoint label.
    """
    sources, onsets, families, codes = [], [], set(), {}
    for path in sorted(folder.glob('*failure_status-*.csv')):
        d = read_topic(path, '%time')
        if 'field.data' not in d:
            raise ValueError(f'missing fault status in {path}')
        v = pd.to_numeric(d['field.data'], errors='coerce').to_numpy()
        if not np.isfinite(v).all() or not np.isin(v, [0, 1, 2, 3]).all():
            raise ValueError(f'unverified fault code in {path}')
        active = v > 0
        if active.any():
            raw_family = path.stem.split('failure_status-')[-1]
            family = {'engines': 'engine', 'elevator': 'elev'}.get(raw_family, raw_family)
            families.add(family)
            onsets.append(int(d.loc[active, '%time'].iloc[0]))
        sources.append(path)
        codes[path.name] = sorted(set(v.astype(int).tolist()))
    if 'no_ground_truth' in folder.name:
        return {'eligible': False, 'reason': 'no_ground_truth', 'sources': sources}
    if not onsets and 'no_failure' not in folder.name:
        return {'eligible': False, 'reason': 'missing_verified_fault_onset', 'sources': sources}
    if onsets and 'no_failure' in folder.name:
        return {'eligible': False, 'reason': 'nominal_filename_conflicts_with_fault_topic', 'sources': sources}
    return {'eligible': True, 'onset_ns': min(onsets) if onsets else None,
            'families': sorted(families), 'codes': codes, 'sources': sources}


def rfly_fault_flags(log: Path):
    """Return the native-clock injection-control flags, not spreadsheet offsets.

    Author toolkit fileprocess.py RFLY_fault_id_check defines fault ranges
    123450..123459 / 123540..123549 and default IDs 0/1500. These label logged
    injection state, not physical severity or a guarantee of actual damage.
    """
    paths = sorted(log.glob('*_rfly_ctrl_lxl_0.csv'))
    if len(paths) != 1:
        raise ValueError('missing or ambiguous injection-control topic')
    d = read_topic(paths[0], 'timestamp')
    ids = pd.to_numeric(d['id'], errors='coerce').to_numpy()
    active = ((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))
    if not (active | np.isin(ids, [0, 1500])).all():
        raise ValueError('unverified injection-control ID')
    return d, active, paths[0]
