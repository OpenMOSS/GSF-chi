from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gsf_chi.batching import SampleDataset, collate
from gsf_chi.config import parse_config
from gsf_chi.controls import attach_gsf_support_masks
from gsf_chi.data.axial import load_acmp, verify_dataset
from gsf_chi.model import MolecularModel
from gsf_chi.paths import ROOT
from gsf_chi.training import set_seed


@dataclass
class Metrics:
    loss: float
    accuracy: float
    pair_flip_rate: float
    pair_logit_gap: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_loss, total, correct = (0.0, 0, 0)
    records: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch)
        labels = batch["label"]
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        score = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        for base, y, p, s in zip(
            batch["base_id"].cpu().tolist(), labels.cpu().tolist(), pred.cpu().tolist(), score
        ):
            records[int(base)].append((int(y), int(p), float(s)))
    pairs = [v for v in records.values() if len(v) == 2]
    flip = np.mean([v[0][1] != v[1][1] for v in pairs]) if pairs else float("nan")
    gap = np.mean([abs(v[0][2] - v[1][2]) for v in pairs]) if pairs else float("nan")
    return Metrics(total_loss / total, correct / total, float(flip), float(gap))


def train_one(
    mode: str, seed: int, samples: list[dict], split: dict, args: argparse.Namespace
) -> dict:
    set_seed(seed)
    device = torch.device(args.device)
    loaders = {}
    for name, key, shuffle in [
        ("train", "train_index", True),
        ("val", "val_index", False),
        ("test", "test_index", False),
    ]:
        loaders[name] = DataLoader(
            SampleDataset(samples, split[key]),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=0,
        )
    model = MolecularModel(
        mode=mode,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        chiral_scale_init=args.gsf_scale,
        gsf_scope=getattr(args, "gsf_scope", "global"),
        condition_unit_axis=not getattr(args, "fixed_gsf_axis", False),
        condition_unit_frequency=not getattr(args, "fixed_gsf_frequency", False),
        learn_pair_gate=not getattr(args, "uniform_gsf_gate", False),
        use_phase_initialization=not getattr(args, "zero_gsf_phase_init", False),
        gsf_support_mode="fixed"
        if getattr(args, "gsf_support_mode", "full") in {"random", "distance"}
        else getattr(args, "gsf_support_mode", "full"),
        gsf_support_fraction=getattr(args, "gsf_support_fraction", 0.0),
        gsf_rotary_mode=getattr(args, "gsf_rotary_mode", "residual"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr / 20
    )
    best = {"epoch": 0, "val_loss": float("inf"), "state": None}
    history = []
    history_path = args.output.with_name(f"{args.output.stem}.{mode}.seed{seed}.history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch), batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        val = evaluate(model, loaders["val"], device)
        epoch_record = {"epoch": epoch, "validation": val.__dict__}
        epoch_test = None
        if getattr(args, "record_test_history", False):
            epoch_test = evaluate(model, loaders["test"], device)
            epoch_record["test"] = epoch_test.__dict__
        history.append(epoch_record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            handle.flush()
        if val.loss < best["val_loss"]:
            best = {
                "epoch": epoch,
                "val_loss": val.loss,
                "state": copy.deepcopy(model.state_dict()),
            }
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_m = evaluate(model, loaders["train"], device)
            print(
                f"mode={mode:5s} seed={seed} epoch={epoch:03d} train={train_m.accuracy:.3f} val={val.accuracy:.3f}"
                + (
                    f" test={epoch_test.accuracy:.3f} pair_flip={epoch_test.pair_flip_rate:.3f}"
                    if epoch_test is not None
                    else ""
                ),
                flush=True,
            )
    model.load_state_dict(best["state"])
    metrics = {name: evaluate(model, loader, device).__dict__ for name, loader in loaders.items()}
    checkpoint_path = None
    if getattr(args, "save_checkpoint", False):
        checkpoint_path = args.output.with_name(f"{args.output.stem}.{mode}.seed{seed}.best.pt")
        torch.save(
            {
                "mode": mode,
                "seed": seed,
                "best_epoch": best["epoch"],
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": best["epoch"],
        "parameter_count": sum((p.numel() for p in model.parameters())),
        "metrics": metrics,
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--annotation-source",
        choices=["curated", "automatic"],
        default="curated",
        help="Use the curated ACMP units or the label-independent ChiralFinder detections serialized with the released conformers.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "axial_rotation.json")
    parser.add_argument("--modes", nargs="+", default=["base", "token", "local", "gsf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gsf-scale", type=float, default=5.0)
    parser.add_argument(
        "--gsf-scope",
        choices=["global", "query_anchor", "incident_anchor"],
        default="global",
        help="Restrict the GSF correction to all pairs, anchor queries, or anchor-incident pairs.",
    )
    parser.add_argument(
        "--gsf-support-mode",
        choices=["full", "random", "distance", "learned_topk"],
        default="full",
        help="Training-time all-pair support. Random and distance supports are fixed parity-even masks; learned_topk selects by the learned gate.",
    )
    parser.add_argument(
        "--gsf-support-fraction",
        type=float,
        default=0.0,
        help="Fraction of directed atom pairs retained by sparse support; zero matches each molecule/unit's query-anchor entry count.",
    )
    parser.add_argument("--gsf-support-seed", type=int, default=20260828)
    parser.add_argument(
        "--gsf-rotary-mode",
        choices=["residual", "direct_replace"],
        default="residual",
        help="Use the residual rotary correction or directly replace the selected heads' ordinary query-key score by q^T R k.",
    )
    parser.add_argument("--fixed-gsf-axis", action="store_true")
    parser.add_argument("--fixed-gsf-frequency", action="store_true")
    parser.add_argument("--uniform-gsf-gate", action="store_true")
    parser.add_argument("--zero-gsf-phase-init", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.set_defaults(record_test_history=False)
    parser.add_argument("--log-every", type=int, default=5)
    return parse_config(parser)


def main() -> None:
    args = parse_args()
    samples, split = load_acmp(args.data_dir, annotation_source=args.annotation_source)
    audit = verify_dataset(samples, split)
    support_audit = None
    if args.gsf_support_mode in {"random", "distance"}:
        support_audit = attach_gsf_support_masks(
            samples,
            mode=args.gsf_support_mode,
            fraction=args.gsf_support_fraction,
            support_seed=args.gsf_support_seed,
        )
        print("support_audit=" + json.dumps(support_audit), flush=True)
    print("dataset_audit=" + json.dumps(audit, ensure_ascii=False), flush=True)
    results = []
    for mode in args.modes:
        for seed in args.seeds:
            results.append(train_one(mode, seed, samples, split, args))
    payload = {
        "selection_contract": "minimize validation loss; the test split is not evaluated during model selection; reported test metrics are evaluated after restoring the validation-selected checkpoint",
        "args": dict(vars(args)),
        "dataset_audit": audit,
        "support_audit": support_audit,
        "results": results,
    }
    payload["args"]["data_dir"] = str(payload["args"]["data_dir"])
    payload["args"]["output"] = str(payload["args"]["output"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
