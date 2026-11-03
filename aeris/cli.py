"""AERIS reproduction commands; all generated artifacts stay in local runs/."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data.tvt_raw import sha256
from src.models.aeris_horizon_frozen import FrozenHorizonAERIS
from .protocol import SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Causal preprocessing of the public CSV exports")
    p.add_argument("--dataset", choices=SPLITS, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for command in ("train", "fit-reference", "evaluate", "reproduce"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=SPLITS, required=True)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--setting", choices=sorted(set(sum((list(v) for v in SPLITS.values()), []))))
        if command == "reproduce":
            p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
            p.add_argument("--out", type=Path, default=Path("runs"))
        else:
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score", help="Score raw-unit complete windows with a locally fitted monitor")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        from .data import prepare
        return prepare(args.dataset, args.raw, args.out)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no implicit CPU fallback")
    if args.command == "score":
        side = args.output.with_suffix(".provenance.json")
        if args.output.suffix != ".npz" or args.output.exists() or side.exists():
            raise ValueError("Choose an unused .npz output path")
        provenance = json.loads(args.bundle.with_name("provenance.json").read_text())
        if sha256(args.bundle) != provenance["bundle_sha256"]:
            raise ValueError("Bundle checksum mismatch")
        with np.load(args.input, allow_pickle=False) as d:
            windows, groups = d["windows"].copy(), d["source_groups"].copy()
        monitor = FrozenHorizonAERIS.load(args.bundle, args.device)
        values = monitor.score_windows(windows, groups, batch_size=args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **values)
        side.write_text(json.dumps(dict(input_sha256=sha256(args.input),
             bundle_sha256=sha256(args.bundle), output_sha256=sha256(args.output),
             windows=len(windows), device=args.device, calibration_updated=False), indent=2) + "\n")
        return print(args.output)
    from .pipeline import train_base, fit_reference, evaluate
    if args.setting and args.setting not in SPLITS[args.dataset]:
        parser.error("Setting does not belong to the selected dataset")
    if args.command != "reproduce" and args.setting is None:
        parser.error("--setting is required for this command")
    if args.command == "train":
        return train_base(args.data_root, args.dataset, args.setting, args.seed, args.out, args.device)
    if args.command == "fit-reference":
        return fit_reference(args.data_root, args.dataset, args.setting, args.out, args.device)
    if args.command == "evaluate":
        return evaluate(args.data_root, args.dataset, args.setting, args.out, args.device)
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Duplicate seeds are not allowed")
    for setting in ([args.setting] if args.setting else SPLITS[args.dataset]):
        for seed in args.seeds:
            output = args.out / args.dataset / setting / f"seed{seed}"
            train_base(args.data_root, args.dataset, setting, seed, output, args.device)
            fit_reference(args.data_root, args.dataset, setting, output, args.device)
            evaluate(args.data_root, args.dataset, setting, output, args.device)
