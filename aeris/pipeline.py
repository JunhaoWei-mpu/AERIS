"""Training, nominal reference fitting, and evaluation of frozen AERIS equations.

The numerical modules under src/ are copied unchanged from the research code.
This module replaces development-runner dependencies with a portable workflow.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.tvt_raw import sha256
from src.data.tvt_dataset import TVTWindowDataset, load_tvt_flight
from src.data.flight_windows import Standardizer
from src.models.tvt_control_chain import ChainInputs, SEMANTICS
from src.models.tvt_horizon_response import summarize_horizon_response
from src.models.tvt_matched_chain import MatchedChainReference
from src.models.tvt_response_median import ay_energy
from src.models.tvt_horizon_monitor import HorizonMonitorCalibration, pooled_evidence
from src.models.tvt_online import make_window_independent
from src.models.tvt_stable_window_inference import StableWindowInference
from src.models.aeris_horizon_frozen import FrozenHorizonAERIS
from src.eval.vus_metrics import vus_roc_pr_multiseries
from src.utils.seed import set_seed
from .protocol import ROOT, prepare_nominal, nominal_library, development_roles, common_window_support, read_split
from .training import Channels, IndependentValidationTrainer, build_base, loader, reference_subset


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def train_base(root, dataset, setting, seed, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(seed, deterministic=True)
    split, scaler, train, phases, meta, constant, lineage = prepare_nominal(root, dataset, setting)
    dump(output / "fit_manifest.json", dict(dataset=dataset, setting=setting, seed=seed,
         normalization=scaler.as_dict(), constant=constant,
         base_config_sha256=sha256(ROOT / "configs/base.json"),
         method_config_sha256=sha256(ROOT / "configs/method.json"), **lineage))
    model, cfg = build_base({"backbone_config": str(ROOT / "configs/base.json")}, dataset)
    cfg["seed"] = seed
    model.to(device)
    ds = Channels(train, SEMANTICS[dataset].base)
    gate, ids = reference_subset(ds, 8064, 30000 + seed)
    with torch.no_grad():
        model.tn_branch_hard_gate.calibrate(torch.cat([x.std(1) for x in loader(gate, 256)]).to(device))
    np.save(output / "training_gate_members.npy", ids)
    trainer = IndependentValidationTrainer(model, loader(ds, 64, 20000 + seed),
        loader(Channels(phases["selection"], SEMANTICS[dataset].base), 256),
        None, output / "backbone", cfg, device)
    dump(output / "checkpoint_selection.json", trainer.train())
    model.load_state_dict(torch.load(output / "backbone/best.pth", map_location=device, weights_only=False)["model"])
    torch.save(model.cpu().eval(), output / "base.pth")
    return output


@torch.no_grad()
def summarize(ds, inputs, split, constant):
    rows = [summarize_horizon_response(x, inputs, constant) for x in loader(ds, 512)]
    data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    data["groups"] = np.array([split["flight_to_group"][ds.flights[r.flight_index].flight_id]
                               for r in ds.refs])
    return data


def statistics(reference, query):
    out = {}
    for mode, prefix in (("context", ""), ("history", "zero_")):
        out.update({prefix + k: v for k, v in reference.evidence(query, mode).items()})
        _, out[prefix + "ay_L"], _ = ay_energy(reference, query, mode)
    out.update(pooled_evidence(reference, query))
    return out


@torch.no_grad()
def base_scores(ds, model, constant, dataset, device, batch_size=128):
    indices = list(SEMANTICS[dataset].base)
    active = [i for i, j in enumerate(indices) if j not in constant]
    result = []
    for x in loader(ds, batch_size):
        x = x.to(device)
        x[:, :, constant] = 0
        y = x[:, :, indices]
        result.append((model(y)["recon"] - y).square()[:, :, active].mean((1, 2)).cpu().numpy())
    return np.concatenate(result)


def fit_reference(root, dataset, setting, output, device):
    output = Path(output)
    if (output / "monitor.pth").exists() or (output / "calibration_failure.json").exists():
        raise FileExistsError("Refusing to overwrite fitted reference/calibration")
    fit = json.loads((output / "fit_manifest.json").read_text())
    split, scaler, train, phases, meta, constant, lineage = prepare_nominal(root, dataset, setting)
    if (fit["dataset"], fit["setting"]) != (dataset, setting):
        raise ValueError("Checkpoint platform/setting mismatch")
    if fit["data_hashes"] != lineage["data_hashes"] or fit["split_sha256"] != lineage["split_sha256"]:
        raise ValueError("Nominal data or split changed after Base fitting")
    if fit["normalization"] != scaler.as_dict() or fit["constant"] != constant:
        raise ValueError("Nominal normalization changed")
    if fit["base_config_sha256"] != sha256(ROOT / "configs/base.json") or fit["method_config_sha256"] != sha256(ROOT / "configs/method.json"):
        raise ValueError("Configuration changed after Base fitting")
    base = torch.load(output / "base.pth", map_location=device, weights_only=False)
    inference = StableWindowInference(make_window_independent(base.eval()))
    inputs = ChainInputs(dataset, scaler.as_dict(), .4)
    libds = nominal_library(train, scaler, split, meta)
    reference = MatchedChainReference().fit(summarize(libds, inputs, split, constant))
    data = {}
    for phase in ("regression", "tail", "development"):
        ds = phases[phase]
        d = statistics(reference, summarize(ds, inputs, split, constant))
        d["base"] = base_scores(ds, inference, constant, dataset, device)
        if phase == "development":
            role = development_roles(ds, split, meta, dataset)
            for key, value in (("branch", 0), ("final", 1)):
                data[key] = {k: v[role == value] for k, v in d.items()}
        else:
            valid = common_window_support(ds)
            data[phase] = {k: v[valid] for k, v in d.items()}
    try:
        calibration = HorizonMonitorCalibration().fit(data["regression"], data["tail"], data["branch"], data["final"])
    except (ValueError, AssertionError) as exc:
        dump(output / "calibration_failure.json", dict(error=str(exc), test_files_opened=False,
             reference_rows=len(reference.library["groups"]), fallback=False))
        raise
    bundle = dict(format="aeris-horizon35-v1", platform=dataset, normalization=scaler.as_dict(),
                  constant=constant, reference=reference, calibration=calibration,
                  base=torch.load(output / "base.pth", map_location="cpu", weights_only=False).eval())
    torch.save(bundle, output / "monitor.pth")
    dump(output / "provenance.json", dict(platform=dataset, split=setting, seed=fit["seed"],
         bundle_sha256=sha256(output / "monitor.pth"), base_sha256=sha256(output / "base.pth"),
         reference_rows=len(reference.library["groups"]),
         nominal_role_counts={k: len(v["base"]) for k, v in data.items()},
         normalization_updated=False, test_files_opened=False, model_selection=False))


def contiguous_runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def evaluate(root, dataset, setting, output, device):
    output = Path(output)
    dest = output / "evaluation"
    dest.mkdir(exist_ok=False)
    bundle_path = output / "monitor.pth"
    provenance = json.loads((output / "provenance.json").read_text())
    if sha256(bundle_path) != provenance["bundle_sha256"]:
        raise ValueError("Bundle checksum mismatch")
    monitor = FrozenHorizonAERIS.load(bundle_path, device)
    if monitor.platform != dataset or provenance["split"] != setting:
        raise ValueError("Evaluation platform/setting differs from fitted bundle")
    split, _ = read_split(dataset, setting)
    meta = {f["flight_id"]: f for f in json.loads((Path(root) / "meta.json").read_text())["flights"]}
    flights = []
    for name in split["test"]:
        path = Path(root) / (name + ".npz")
        if sha256(path) != meta[name]["npz_sha256"]:
            raise ValueError("Test-file checksum mismatch: " + name)
        flights.append(load_tvt_flight(path))
    st = monitor.normalization
    scaler = Standardizer(np.array(st["mean"], np.float32), np.array(st["std"], np.float32), np.array(st["count"]))
    ds = TVTWindowDataset(flights, split["test"], scaler, 100, 1, False)
    query = summarize(ds, monitor.inputs, split, monitor.constant)
    d = statistics(monitor.reference, query)
    d["base"] = base_scores(ds, monitor.base, monitor.constant, dataset, device)
    scores = monitor.calibration.score(d)
    scores["base"] = d["base"]
    support = common_window_support(ds)
    methods = list(scores)
    labels, series = [], {key: [] for key in methods}
    for i, flight in enumerate(ds.flights):
        ids = np.array([j for j, ref in enumerate(ds.refs) if ref.flight_index == i])
        ends = np.array([ds.refs[j].end - 1 for j in ids])
        common = np.zeros(flight.n_timesteps, bool)
        common[ends] = support[ids]
        common &= np.isin(flight.label, [0, 1])
        trace = dict(time=flight.t, label=flight.label, common=common)
        for key in methods:
            values = np.full(flight.n_timesteps, np.nan)
            values[ends] = scores[key][ids]
            trace[key] = values
        np.savez_compressed(dest / (flight.flight_id + ".npz"), **trace)
        for lo, hi in contiguous_runs(common):
            labels.append(flight.label[lo:hi])
            for key in methods:
                series[key].append(trace[key][lo:hi])
    y = np.concatenate(labels)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation needs both classes on common support")
    metrics = {}
    for key, ss in series.items():
        vr, vp = vus_roc_pr_multiseries(labels, ss, max_buffer=100, thresholds=250)
        flat = np.concatenate(ss)
        metrics[key] = dict(vus_roc=float(vr), vus_pr=float(vp),
                            auroc=float(roc_auc_score(y, flat)), aupr=float(average_precision_score(y, flat)))
    dump(dest / "metrics.json", dict(dataset=dataset, setting=setting,
         seed=provenance["seed"], common_samples=len(y), methods=metrics,
         comparison_support_samples=199, model_input_samples=100))
    print(json.dumps(metrics["full"], indent=2))
