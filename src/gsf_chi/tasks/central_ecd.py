from __future__ import annotations

import argparse
import json
from pathlib import Path

import gsf_chi.paths as paths
import gsf_chi.tasks.axial_ecd as tasks_axial_ecd
from gsf_chi.config import parse_config
from gsf_chi.data.central_ecd import load_dataset
from gsf_chi.splits import PAIR_SAFE_SPLIT, SPLIT_PROTOCOLS, apply_split_protocol


def parse_args() -> argparse.Namespace:
    root = paths.ROOT / "data" / "central_ecd_raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=root / "extracted" / "ECD")
    parser.add_argument(
        "--graph-path", type=Path, default=root / "ecd_column_charity_new_smiles.npy"
    )
    parser.add_argument("--cache", type=Path, default=root / "gsf_central_ecd_cache.pkl")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--split-protocol",
        choices=SPLIT_PROTOCOLS,
        default=PAIR_SAFE_SPLIT,
        help="pair-safe ChiDeK split or ECDFormer's sample-level 90/5/5 split",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1101,
        help="NumPy shuffle seed for the ECDFormer 90/5/5 protocol",
    )
    parser.add_argument("--output", type=Path, default=paths.ROOT / "outputs" / "central_ecd.json")
    parser.add_argument("--modes", nargs="+", default=["token", "gsf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Split training batches for gradient accumulation and bound eval batches.",
    )
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-chiral-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gsf-scale", type=float, default=5.0)
    parser.add_argument(
        "--gsf-scope", choices=["global", "query_anchor", "incident_anchor"], default="global"
    )
    parser.add_argument("--fixed-gsf-axis", action="store_true")
    parser.add_argument("--fixed-gsf-frequency", action="store_true")
    parser.add_argument("--share-gsf-axis-frequency", action="store_true")
    parser.add_argument("--uniform-gsf-gate", action="store_true")
    parser.add_argument("--zero-gsf-phase-init", action="store_true")
    parser.add_argument("--chiral-logit-reg", type=float, default=0.0)
    parser.add_argument("--field-smoothness-reg", type=float, default=0.0)
    parser.add_argument("--stereo-unit-dropout", type=float, default=0.0)
    parser.add_argument(
        "--readout",
        choices=[
            "legacy",
            "parity",
            "parity_split",
            "parity_aux",
            "parity_chain",
            "parity_split_chain",
        ],
        default="parity",
        help="parity_split gives Symbol a private parity-even position-context head, isolating its gradient from the supervised Position head.",
    )
    parser.add_argument(
        "--number-readout", choices=["categorical", "ordinal"], default="categorical"
    )
    parser.add_argument(
        "--position-decode",
        choices=["independent", "monotonic", "posterior_mean"],
        default="independent",
        help="Peak-bin argmax, monotonic sequence, or rounded posterior mean.",
    )
    parser.add_argument("--position-decode-temperature", type=float, default=1.0)
    parser.add_argument("--scheduler-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--number-loss-weight", type=float, default=1.0)
    parser.add_argument("--number-ordinal-weight", type=float, default=0.0)
    parser.add_argument("--number-risk-weight", type=float, default=0.0)
    parser.add_argument("--selection-number-weight", type=float, default=0.0)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--position-ordinal-weight", type=float, default=0.0)
    parser.add_argument("--position-risk-weight", type=float, default=0.0)
    parser.add_argument("--position-monotonic-weight", type=float, default=0.0)
    parser.add_argument("--position-loss-reduction", choices=["peak", "molecule"], default="peak")
    parser.add_argument(
        "--position-risk-reduction", choices=["peak_mse", "molecule_rmse"], default="peak_mse"
    )
    parser.add_argument("--position-soft-label-sigma", type=float, default=0.0)
    parser.add_argument("--selection-position-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-objective",
        choices=["composite", "position", "scaled_minimum"],
        default="composite",
    )
    parser.add_argument("--selection-symbol-target", type=float, default=0.533)
    parser.add_argument("--selection-symbol-scale", type=float, default=0.006)
    parser.add_argument("--selection-number-target", type=float, default=1.01)
    parser.add_argument("--selection-number-scale", type=float, default=0.09)
    parser.add_argument("--selection-position-target", type=float, default=2.02)
    parser.add_argument("--selection-position-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parse_config(parser)


def main() -> None:
    args = parse_args()
    samples, cached_split, audit = load_dataset(args)
    split, split_contract = apply_split_protocol(
        samples, cached_split, args.split_protocol, args.split_seed
    )
    audit = dict(audit)
    audit["data_split_contract"] = split_contract
    audit["split_molecules"] = {name: len(indices) for name, indices in split.items()}
    print("dataset_audit=" + json.dumps(audit, sort_keys=True), flush=True)
    train_function = tasks_axial_ecd.train_one
    results = [
        train_function(mode, seed, samples, split, args)
        for mode in args.modes
        for seed in args.seeds
    ]
    arg_dict = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    payload = {
        "benchmark": "CMCDS public reproducible subset",
        "selection_contract": (
            "maximize the minimum scaled validation margin across Symbol, Number, and Position"
            if args.selection_objective == "scaled_minimum"
            else f"maximize validation Symbol accuracy minus {args.selection_number_weight:g}*Number_RMSE minus {args.selection_position_weight:g}*Position_RMSE"
        )
        + ", with validation-loss tie-break; the test split is evaluated once after restoring the validation-selected checkpoint",
        "args": arg_dict,
        "dataset_audit": audit,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
