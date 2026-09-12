from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gsf_chi.model as models
import gsf_chi.training as training
from gsf_chi.config import parse_config
from gsf_chi.data.rs import (
    CACHE_FORMAT,
    ShardedDataset,
    ShardShuffleSampler,
    _atomic_json,
    _resolve_inputs,
    collate,
    prepare_split,
)
from gsf_chi.paths import ROOT


@dataclass
class Metrics:
    loss: float
    conformer_accuracy: float
    stereoisomer_accuracy: float
    pair_complement_rate: float
    complete_pairs: int


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    stereo_scores: dict[str, list[float]] = defaultdict(list)
    stereo_labels: dict[str, int] = {}
    stereo_bases: dict[str, str] = {}
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        logits = model(batch)
        labels = batch["label"]
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)
        scores = (logits[:, 1] - logits[:, 0]).detach().cpu().tolist()
        for stereo_id, base_key, label, score in zip(
            batch["stereo_id"], batch["base_key"], labels.cpu().tolist(), scores
        ):
            if stereo_id in stereo_labels and stereo_labels[stereo_id] != int(label):
                raise ValueError(f"inconsistent labels for stereoisomer {stereo_id}")
            stereo_scores[stereo_id].append(float(score))
            stereo_labels[stereo_id] = int(label)
            stereo_bases[stereo_id] = base_key
    stereo_predictions = {key: int(np.mean(values) > 0.0) for key, values in stereo_scores.items()}
    stereo_correct = sum(
        (stereo_predictions[key] == stereo_labels[key] for key in stereo_predictions)
    )
    by_base: dict[str, list[str]] = defaultdict(list)
    for stereo_id, base_key in stereo_bases.items():
        by_base[base_key].append(stereo_id)
    complete = [ids for ids in by_base.values() if len(ids) == 2]
    complement = (
        float(
            np.mean([stereo_predictions[ids[0]] != stereo_predictions[ids[1]] for ids in complete])
        )
        if complete
        else float("nan")
    )
    return Metrics(
        loss=total_loss / max(total, 1),
        conformer_accuracy=correct / max(total, 1),
        stereoisomer_accuracy=stereo_correct / max(len(stereo_predictions), 1),
        pair_complement_rate=complement,
        complete_pairs=len(complete),
    )


def _make_loaders(cache_dir: Path, batch_size: int, seed: int) -> tuple[dict, ShardShuffleSampler]:
    datasets = {
        split: ShardedDataset(cache_dir / split) for split in ("train", "validation", "test")
    }
    sampler = ShardShuffleSampler(datasets["train"], seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate,
            num_workers=0,
            pin_memory=False,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        ),
    }
    return (loaders, sampler)


def _epoch_history_path(output: Path, mode: str, seed: int) -> Path:
    return output.with_name(f"{output.stem}.{mode}.seed{seed}.history.jsonl")


def _append_epoch_history(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def train_one(mode: str, seed: int, args: argparse.Namespace) -> dict:
    training.set_seed(seed)
    device = torch.device(args.device)
    loaders, sampler = _make_loaders(args.cache_dir, args.batch_size, seed)
    model = models.MolecularModel(
        mode=mode,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        chiral_scale_init=args.gsf_scale,
        stereo_unit_dropout=args.stereo_unit_dropout,
        n_chiral_heads=args.n_chiral_heads,
        gsf_scope=args.gsf_scope,
        condition_unit_axis=not args.fixed_gsf_axis,
        condition_unit_frequency=not args.fixed_gsf_frequency,
        learn_pair_gate=not args.uniform_gsf_gate,
        use_phase_initialization=not args.zero_gsf_phase_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr / 20.0
    )
    best = {"epoch": 0, "val_loss": float("inf"), "state": None, "validation": None}
    history = []
    history_path = _epoch_history_path(args.output, mode, seed)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * len(batch["label"])
            seen += len(batch["label"])
            if args.log_steps > 0 and step % args.log_steps == 0:
                print(
                    f"mode={mode} seed={seed} epoch={epoch} step={step}/{len(loaders['train'])} train_loss={running_loss / seen:.5f}",
                    flush=True,
                )
        scheduler.step()
        validation = evaluate(model, loaders["validation"], device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "validation": asdict(validation),
        }
        history.append(epoch_record)
        _append_epoch_history(history_path, epoch_record)
        print(
            f"mode={mode} seed={seed} epoch={epoch} train_loss={running_loss / max(seen, 1):.5f} val_loss={validation.loss:.5f} val_acc={validation.conformer_accuracy:.4f}",
            flush=True,
        )
        if validation.loss < best["val_loss"]:
            best = {
                "epoch": epoch,
                "val_loss": validation.loss,
                "validation": asdict(validation),
                "state": copy.deepcopy(model.state_dict()),
            }
    if best["state"] is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best.pop("state"))
    test = evaluate(model, loaders["test"], device)
    return {
        "mode": mode,
        "seed": seed,
        "parameter_count": sum((parameter.numel() for parameter in model.parameters())),
        "best": best,
        "test": asdict(test),
        "history_path": str(history_path),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pkl", type=Path, default=ROOT / "data" / "RS_train.pkl")
    parser.add_argument("--val-pkl", type=Path, default=ROOT / "data" / "RS_validation.pkl")
    parser.add_argument("--test-pkl", type=Path, default=ROOT / "data" / "RS_test.pkl")
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "gsf_official_rs_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rs.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--prepare-log-every", type=int, default=5000)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--modes", nargs="+", choices=["base", "token", "local", "gsf"], default=["token", "gsf"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-chiral-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stereo-unit-dropout", type=float, default=0.0)
    parser.add_argument("--gsf-scale", type=float, default=5.0)
    parser.add_argument(
        "--gsf-scope", choices=["global", "query_anchor", "incident_anchor"], default="global"
    )
    parser.add_argument("--fixed-gsf-axis", action="store_true")
    parser.add_argument("--fixed-gsf-frequency", action="store_true")
    parser.add_argument("--uniform-gsf-gate", action="store_true")
    parser.add_argument("--zero-gsf-phase-init", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-steps", type=int, default=250)
    return parse_config(parser)


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.rebuild_cache:
        for split in ("train", "validation", "test"):
            index = args.cache_dir / split / "index.json"
            if index.exists():
                index.unlink()
    inputs = _resolve_inputs(args)
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_val_samples,
        "test": args.max_test_samples,
    }
    cache_metadata = {
        split: prepare_split(
            split, source, args.cache_dir, args.shard_size, limits[split], args.prepare_log_every
        )
        for split, source in inputs.items()
    }
    base_sets = {split: set(metadata["base_keys"]) for split, metadata in cache_metadata.items()}
    overlap = {
        "train:validation": len(base_sets["train"] & base_sets["validation"]),
        "train:test": len(base_sets["train"] & base_sets["test"]),
        "validation:test": len(base_sets["validation"] & base_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"molecule leakage across official splits: {overlap}")
    audit = {
        "cache_format": CACHE_FORMAT,
        "official_splits_preserved": True,
        "split_base_overlap": overlap,
        "splits": {
            split: {
                key: value for key, value in metadata.items() if key not in {"base_keys", "shards"}
            }
            for split, metadata in cache_metadata.items()
        },
    }
    print("dataset_audit=" + json.dumps(audit, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    results = [train_one(mode, seed, args) for mode in args.modes for seed in args.seeds]
    payload = {
        "benchmark": "official ChIRo R/S conformer splits",
        "selection_contract": "best validation loss; the test split is evaluated once after restoring the validation-selected checkpoint",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset_audit": audit,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
