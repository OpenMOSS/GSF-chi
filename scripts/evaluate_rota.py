"""Evaluate saved axial models on prepared RotA conformers."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gsf_chi.batching import collate
from gsf_chi.checkpoints import load_model
from gsf_chi.data.ecd import collate_ecd
from gsf_chi.ecd_metrics import evaluate as evaluate_ecd
from gsf_chi.tasks.axial_rotation import evaluate as evaluate_rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Training result JSON containing arguments and checkpoint paths.",
    )
    parser.add_argument("--task", choices=["axial_ecd", "axial_rotation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with args.prepared.open("rb") as handle:
        data = pickle.load(handle)
    run = json.loads(args.run.read_text())
    is_ecd = args.task == "axial_ecd"
    loader = DataLoader(
        data["gsf_samples"],
        batch_size=args.batch_size,
        collate_fn=collate_ecd if is_ecd else collate,
        shuffle=False,
    )
    evaluate = evaluate_ecd if is_ecd else evaluate_rotation
    results = []
    for row in run["results"]:
        path = row.get("checkpoint_path")
        if path is None:
            raise ValueError("The run has no checkpoint; train with --save-checkpoint.")
        model = load_model(Path(path), args.task, args.device, run["args"])
        results.append(
            {
                "seed": row["seed"],
                "checkpoint": path,
                "metrics": asdict(evaluate(model, loader, torch.device(args.device))),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"task": args.task, "audit": data["audit"], "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
