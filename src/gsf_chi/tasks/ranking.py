from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gsf_chi.training as training
from gsf_chi.config import parse_config
from gsf_chi.data.ranking import (
    CACHE_FORMAT,
    PairBatchSampler,
    ShardedPairDataset,
    _atomic_json,
    collate_pairs,
    prepare_split,
)
from gsf_chi.model import RankingModel
from gsf_chi.paths import ROOT
from gsf_chi.training import ModelEMA


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _epoch_history_path(output: Path, mode: str, seed: int) -> Path:
    return output.with_name(f"{output.stem}.{mode}.seed{seed}.history.jsonl")


def _append_epoch_history(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _mirror_chirality(batch: dict) -> dict:
    """Flip handedness while preserving every parity-even molecular input."""
    if "chi" not in batch:
        raise ValueError("chirality mirroring requires chi")
    result = dict(batch)
    result["chi"] = -batch["chi"]
    if "local_chi" in batch:
        result["local_chi"] = -batch["local_chi"]
    return result


def _swap_pair_targets(labels: torch.Tensor) -> torch.Tensor:
    """Return the counterfactual labels after flipping each enantiomer."""
    if labels.ndim != 1 or len(labels) % 2:
        raise ValueError("counterfactual chirality requires flat complete pairs")
    return labels.reshape(-1, 2).flip(1).reshape(-1)


@dataclass
class Metrics:
    loss: float
    mse: float
    rmse: float
    ranking_loss: float
    ranking_accuracy: float
    pairs: int


def objective(output, labels, margin, lambda_mse, lambda_rank):
    """MSE on docking scores plus an enantiomer-pair margin loss."""
    if len(labels) % 2:
        raise ValueError("Ranking batches must contain complete pairs")
    mse = F.mse_loss(output, labels)
    direction = torch.sign(labels[0::2] - labels[1::2] + 1e-8)
    ranking = F.relu(margin - direction * (output[0::2] - output[1::2])).mean()
    return (lambda_mse * mse + lambda_rank * ranking, mse, ranking)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args) -> Metrics:
    model.eval()
    total_loss = 0.0
    squared_error = 0.0
    ranking_loss = 0.0
    correct = 0
    samples = 0
    pairs = 0
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        output = model(batch)
        labels = batch["label"]
        loss, _, rank = objective(output, labels, args.margin, args.lambda_mse, args.lambda_rank)
        batch_pairs = len(labels) // 2
        total_loss += loss.item() * len(labels)
        squared_error += F.mse_loss(output, labels, reduction="sum").item()
        ranking_loss += rank.item() * batch_pairs
        target_ranking = torch.round(labels[0::2] * 100.0) > torch.round(labels[1::2] * 100.0)
        output_ranking = torch.round(output[0::2] * 100.0) > torch.round(output[1::2] * 100.0)
        correct += (output_ranking == target_ranking).sum().item()
        samples += len(labels)
        pairs += batch_pairs
    mse = squared_error / max(samples, 1)
    return Metrics(
        loss=total_loss / max(samples, 1),
        mse=mse,
        rmse=math.sqrt(mse),
        ranking_loss=ranking_loss / max(pairs, 1),
        ranking_accuracy=correct / max(pairs, 1),
        pairs=pairs,
    )


def _make_loaders(
    cache_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    shuffle_train_pairs: bool = True,
):
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("--batch-size must be a positive even number (samples, not pairs)")
    if num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    train_dataset = ShardedPairDataset(cache_dir / "train")
    train_sampler = PairBatchSampler(
        train_dataset, pairs_per_batch=batch_size // 2, seed=seed, shuffle=shuffle_train_pairs
    )
    datasets = {
        "train": train_dataset,
        "validation": ShardedPairDataset(cache_dir / "validation"),
        "test": ShardedPairDataset(cache_dir / "test"),
    }
    samplers = {
        "train": train_sampler,
        "validation": PairBatchSampler(
            datasets["validation"], pairs_per_batch=batch_size // 2, seed=seed, shuffle=False
        ),
        "test": PairBatchSampler(
            datasets["test"], pairs_per_batch=batch_size // 2, seed=seed, shuffle=False
        ),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_sampler=samplers[split],
            collate_fn=collate_pairs,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        for split, dataset in datasets.items()
    }
    return (loaders, samplers["train"])


def train_one(mode, seed, args):
    training.set_seed(seed)
    device = torch.device(args.device)
    loaders, sampler = _make_loaders(
        args.cache_dir,
        args.batch_size,
        seed,
        num_workers=args.num_workers,
        shuffle_train_pairs=not args.fixed_train_pair_order,
    )
    model = RankingModel(
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
        share_gsf_axis_frequency=args.share_gsf_axis_frequency,
    ).to(device)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr / 20 if args.min_lr is None else args.min_lr
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    best = None
    history = []
    history_path = _epoch_history_path(args.output, mode, seed)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("")
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        seen = 0
        for batch in loaders["train"]:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            output = model(batch)
            loss, _, _ = objective(
                output, batch["label"], args.margin, args.lambda_mse, args.lambda_rank
            )
            regularization = model.regularization_terms()
            if args.mirror_loss_weight:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)
                mirrored_output = model(_mirror_chirality(batch))
                mirrored_loss, _, _ = objective(
                    mirrored_output,
                    _swap_pair_targets(batch["label"]),
                    args.margin,
                    args.lambda_mse,
                    args.lambda_rank,
                )
                weight = args.mirror_loss_weight
                loss = (1 - weight) * loss + weight * mirrored_loss
                mirrored_reg = model.regularization_terms()
                regularization = {
                    k: (1 - weight) * v + weight * mirrored_reg[k]
                    for k, v in regularization.items()
                }
                loss = loss + args.mirror_consistency_weight * F.mse_loss(
                    output, _swap_pair_targets(mirrored_output)
                )
            loss = (
                loss
                + args.chiral_logit_reg * regularization["chiral_logit"]
                + args.field_smoothness_reg * regularization["field_smoothness"]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update()
            total += loss.item() * len(batch["label"])
            seen += len(batch["label"])
        scheduler.step()
        evaluation_model = ema.model if ema is not None else model
        validation = evaluate(evaluation_model, loaders["validation"], device, args)
        record = {"epoch": epoch, "train_loss": total / seen, "validation": asdict(validation)}
        history.append(record)
        _append_epoch_history(history_path, record)
        print(
            f"mode={mode} seed={seed} epoch={epoch} train_loss={total / seen:.5f} val_rank_acc={validation.ranking_accuracy:.4f}",
            flush=True,
        )
        score = -validation.loss if args.selection_metric == "loss" else validation.ranking_accuracy
        if best is None or score > best["selection_value"]:
            best = {
                "epoch": epoch,
                "selection_value": score,
                "selection_metric": args.selection_metric,
                "validation": asdict(validation),
                "evaluation_weights": "ema" if ema is not None else "online",
                "state": copy.deepcopy(evaluation_model.state_dict()),
            }
    if best is None:
        raise ValueError("Training requires at least one epoch")
    model.load_state_dict(best["state"])
    test = evaluate(model, loaders["test"], device, args)
    checkpoint = args.checkpoint_dir / f"ranking_{mode}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        checkpoint,
        {
            "mode": mode,
            "seed": seed,
            "epoch": best["epoch"],
            "evaluation_weights": best["evaluation_weights"],
            "model_state_dict": best.pop("state"),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    return {
        "mode": mode,
        "seed": seed,
        "parameter_count": sum((p.numel() for p in model.parameters())),
        "checkpoint": str(checkpoint),
        "best": best,
        "test": asdict(test),
        "history_path": str(history_path),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pkl", type=Path, default=ROOT / "data" / "ranking_train.pkl")
    parser.add_argument("--val-pkl", type=Path, default=ROOT / "data" / "ranking_validation.pkl")
    parser.add_argument("--test-pkl", type=Path, default=ROOT / "data" / "ranking_test.pkl")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data" / "gsf_official_ranking_cache"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ranking.json")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "outputs" / "gsf_official_ranking_checkpoints"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--pairs-per-shard", type=int, default=1024)
    parser.add_argument("--prepare-log-every", type=int, default=1000)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--conformer-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        choices=["stereo_group", "legacy_chidek"],
        default="stereo_group",
        help="stereo_group uses corrected SMILES_nostereo pairing; legacy_chidek exactly reproduces the released ID.replace('@', '') sampler",
    )
    parser.add_argument(
        "--fixed-train-pair-order",
        action="store_true",
        help="retain the sampler's one-time pair shuffle for every epoch; this matches the released ChiDeK DataLoader(shuffle=False) contract",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["base", "token", "local", "gsf", "gsf_local", "gsf_token"],
        default=["token", "gsf"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="number of samples; must be even"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers per split; pair sampling remains deterministic",
    )
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
    parser.add_argument(
        "--share-gsf-axis-frequency",
        action="store_true",
        help="share the unit encoder, learned SO(3) axes, and frequencies across all GSF layers",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="cosine eta_min; default keeps the historical lr/20 schedule",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer for training from scratch.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="per-step model-weight EMA decay; 0 disables EMA",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--chiral-logit-reg", type=float, default=0.0)
    parser.add_argument("--field-smoothness-reg", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=["loss", "ranking_accuracy"],
        default="loss",
        help="validation-only checkpoint metric; test is never used for strict selection",
    )
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mirror-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the same-conformer reflected Ranking loss.",
    )
    parser.add_argument(
        "--mirror-consistency-weight",
        type=float,
        default=0.0,
        help="MSE weight between original and pair-aligned reflected scores.",
    )
    return parse_config(parser)


def main() -> None:
    args = parse_args()
    if args.pairs_per_shard <= 0:
        raise ValueError("--pairs-per-shard must be positive")
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("--batch-size must be a positive even number")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.chiral_logit_reg < 0.0 or args.field_smoothness_reg < 0.0:
        raise ValueError("GSF regularization weights must be non-negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.min_lr is not None and (not 0.0 <= args.min_lr <= args.lr):
        raise ValueError("--min-lr must lie in [0, --lr]")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must lie in [0, 1)")
    if not 0 <= 0 <= args.epochs:
        raise ValueError("--swa-start-epoch must lie in [0, --epochs]")
    if args.ema_decay > 0.0 and 0 > 0:
        raise ValueError("--ema-decay and --swa-start-epoch are mutually exclusive")
    if args.rebuild_cache:
        for split in ("train", "validation", "test"):
            index = args.cache_dir / split / "index.json"
            if index.exists():
                index.unlink()
    inputs = {"train": args.train_pkl, "validation": args.val_pkl, "test": args.test_pkl}
    limits = {
        "train": args.max_train_pairs,
        "validation": args.max_val_pairs,
        "test": args.max_test_pairs,
    }
    cache_metadata = {
        split: prepare_split(
            split,
            source,
            args.cache_dir,
            args.pairs_per_shard,
            limits[split],
            args.conformer_seed,
            args.prepare_log_every,
            args.sampling_mode,
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
        "pair_preserving": True,
        "conformer_seed": args.conformer_seed,
        "sampling_mode": args.sampling_mode,
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
    selected_by = (
        "minimum validation composite loss"
        if args.selection_metric == "loss"
        else "maximum validation ranking accuracy"
    )
    selection_contract = f"{selected_by}; the test split is evaluated once after restoring the validation-selected checkpoint"
    payload = {
        "benchmark": "official ChIRo enantiomer ranking splits",
        "selection_contract": selection_contract,
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
