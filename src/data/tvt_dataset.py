"""Strict adapters for versioned raw-rebuilt data; legacy contracts unchanged."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from .alfa import ALFAFlight
from .flight_windows import FlightWindowDataset, fit_nominal_standardizer


@dataclass
class TVTFlight(ALFAFlight):
    nominal_eligible: np.ndarray
    channel_age_s: np.ndarray


def load_tvt_flight(path: Path) -> TVTFlight:
    with np.load(path, allow_pickle=False) as d:
        flight=TVTFlight(path.stem,str(d['fault_type']),d['x'].astype(np.float32),
                         d['label'].astype(np.int8),d['t'].astype(np.float64),float(d['sample_rate']),
                         d['channel_names'].tolist(),d['nominal_eligible'].astype(bool),d['channel_age_s'].astype(np.float32))
    n=flight.n_timesteps
    if flight.label.shape!=(n,) or flight.nominal_eligible.shape!=(n,) or flight.channel_age_s.shape!=flight.x.shape:
        raise ValueError('invalid TVT sample metadata shapes')
    if not np.isin(flight.label,[-1,0,1]).all() or np.any(flight.nominal_eligible & (flight.label!=0)):
        raise ValueError('invalid label or nominal eligibility')
    return flight


def fit_tvt_standardizer(flights, flight_ids):
    # Only finite, explicitly nominal-eligible training samples contribute.
    views=[replace(f,label=(~(f.nominal_eligible & np.isfinite(f.x).all(axis=1))).astype(np.int8)) for f in flights]
    return fit_nominal_standardizer(views,flight_ids)


class TVTWindowDataset(FlightWindowDataset):
    def __init__(self, flights, flight_ids, standardizer, window, stride, nominal_only):
        originals={f.flight_id:f for f in flights}
        views=[replace(f,label=(~f.nominal_eligible).astype(np.int8)) for f in flights] if nominal_only else flights
        super().__init__(views,flight_ids,standardizer,window,stride,nominal_only)
        self.flights=[originals[f.flight_id] for f in self.flights]
        prefixes=[np.r_[0,np.cumsum(~np.isfinite(f.x).all(axis=1))] for f in self.flights]
        self.refs=[r for r in self.refs if prefixes[r.flight_index][r.end]==prefixes[r.flight_index][r.start]]
        if not self.refs:raise RuntimeError('no finite, contiguous eligible TVT windows')
