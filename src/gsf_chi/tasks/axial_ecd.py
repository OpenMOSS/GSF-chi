from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import gsf_chi.batching as batching
import gsf_chi.controls as controls
import gsf_chi.paths as paths
import gsf_chi.training as training
from gsf_chi.config import parse_config
from gsf_chi.controls import apply_training_protocol
from gsf_chi.data.ecd import collate_ecd, dataset_audit, load_ecd_samples
from gsf_chi.ecd import ECDModel
from gsf_chi.ecd_metrics import evaluate, loss_terms, validation_selection_score
from gsf_chi.training import ModelEMA


def train_one(
    mode: str, seed: int, samples: list[dict], split: dict, args: argparse.Namespace
) -> dict:
    training.set_seed(seed)
    device = torch.device(args.device)
    loaders = {}
    loader_specs = [
        ("train", "train_index", True),
        ("val", "val_index", False),
        ("test", "test_index", False),
    ]
    for optional_name in ("mirror_completion", "mirror_completion_pair", "validation_mirror"):
        split_name = f"{optional_name}_index"
        if split_name in split:
            loader_specs.append((optional_name, split_name, False))
    for name, split_name, shuffle in loader_specs:
        loader_batch_size = args.batch_size
        if name != "train" and args.micro_batch_size is not None:
            loader_batch_size = args.micro_batch_size
        loaders[name] = DataLoader(
            batching.SampleDataset(samples, split[split_name]),
            batch_size=loader_batch_size,
            shuffle=shuffle,
            collate_fn=collate_ecd,
            num_workers=0,
        )
    model = ECDModel(
        mode=mode,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        chiral_scale_init=args.gsf_scale,
        readout=args.readout,
        number_readout=getattr(args, "number_readout", "categorical"),
        stereo_unit_dropout=args.stereo_unit_dropout,
        n_chiral_heads=args.n_chiral_heads,
        position_decode=getattr(args, "position_decode", "independent"),
        position_decode_temperature=getattr(args, "position_decode_temperature", 1.0),
        gsf_scope=getattr(args, "gsf_scope", "global"),
        condition_unit_axis=not getattr(args, "fixed_gsf_axis", False),
        condition_unit_frequency=not getattr(args, "fixed_gsf_frequency", False),
        learn_pair_gate=not getattr(args, "uniform_gsf_gate", False),
        use_phase_initialization=not getattr(args, "zero_gsf_phase_init", False),
        share_gsf_axis_frequency=getattr(args, "share_gsf_axis_frequency", False),
        gsf_support_mode="fixed"
        if getattr(args, "gsf_support_mode", "full") in {"random", "distance"}
        else getattr(args, "gsf_support_mode", "full"),
        gsf_support_fraction=getattr(args, "gsf_support_fraction", 0.0),
        gsf_rotary_mode=getattr(args, "gsf_rotary_mode", "residual"),
        n_peak_slots=len(samples[0]["peak_position"]),
    )
    model = model.to(device)
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema = ModelEMA(model, ema_decay) if ema_decay > 0.0 else None
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        getattr(args, "scheduler_epochs", None) or args.epochs,
        eta_min=args.lr / 20 if args.min_lr is None else args.min_lr,
    )
    best = {
        "selection_score": -float("inf"),
        "height_accuracy": -1.0,
        "loss": float("inf"),
        "epoch": 0,
        "state": None,
    }
    history = []
    history_path = args.output.with_name(f"{args.output.stem}.{mode}.seed{seed}.history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            full_batch_size = int(batch["peak_num"].shape[0])
            micro_batch_size = args.micro_batch_size or full_batch_size
            for start in range(0, full_batch_size, micro_batch_size):
                end = min(start + micro_batch_size, full_batch_size)
                micro_batch = {}
                for key, value in batch.items():
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and (value.shape[0] == full_batch_size)
                    ):
                        micro_batch[key] = value[start:end]
                    elif isinstance(value, list) and len(value) == full_batch_size:
                        micro_batch[key] = value[start:end]
                    else:
                        micro_batch[key] = value
                predictions = model(micro_batch)
                position_loss_reduction = getattr(args, "position_loss_reduction", "peak")
                position_risk_reduction = getattr(args, "position_risk_reduction", "peak_mse")
                position_soft_label_sigma = getattr(args, "position_soft_label_sigma", 0.0)
                position_loss_weight = getattr(args, "position_loss_weight", 1.0)
                position_ordinal_weight = getattr(args, "position_ordinal_weight", 0.0)
                position_risk_weight = getattr(args, "position_risk_weight", 0.0)
                position_monotonic_weight = getattr(args, "position_monotonic_weight", 0.0)
                position_ordinal_reduction = "peak_mse"
                position_mask_mode = "target"
                loss, _ = loss_terms(
                    predictions,
                    micro_batch,
                    number_loss_weight=getattr(args, "number_loss_weight", 1.0),
                    number_ordinal_weight=getattr(args, "number_ordinal_weight", 0.0),
                    number_risk_weight=getattr(args, "number_risk_weight", 0.0),
                    position_loss_weight=position_loss_weight,
                    position_ordinal_weight=position_ordinal_weight,
                    position_risk_weight=position_risk_weight,
                    position_monotonic_weight=position_monotonic_weight,
                    position_loss_reduction=position_loss_reduction,
                    position_risk_reduction=position_risk_reduction,
                    position_ordinal_reduction=position_ordinal_reduction,
                    position_soft_label_sigma=position_soft_label_sigma,
                    position_mask_mode=position_mask_mode,
                )
                regularization = model.backbone.regularization_terms()
                loss = (
                    loss
                    + 0.0
                    + args.chiral_logit_reg * regularization["chiral_logit"]
                    + args.field_smoothness_reg * regularization["field_smoothness"]
                )
                (loss * ((end - start) / full_batch_size)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update()
        scheduler.step()
        evaluation_model = ema.model if ema is not None else model
        val = evaluate(evaluation_model, loaders["val"], device)
        epoch_record = {
            "epoch": epoch,
            "evaluation_weights": "ema" if ema is not None else "online",
            "validation": asdict(val),
        }
        if getattr(args, "record_test_history", False):
            epoch_test = evaluate(evaluation_model, loaders["test"], device)
            epoch_record["test"] = asdict(epoch_test)
        history.append(epoch_record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            handle.flush()
        selection_score = validation_selection_score(val, args)
        selection_allowed = epoch >= 1
        if selection_allowed and (
            selection_score > best["selection_score"]
            or (selection_score == best["selection_score"] and val.loss < best["loss"])
        ):
            best = {
                "selection_score": selection_score,
                "height_accuracy": val.height_accuracy,
                "loss": val.loss,
                "epoch": epoch,
                "state": copy.deepcopy(evaluation_model.state_dict()),
            }
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            message = f"mode={mode:5s} seed={seed} epoch={epoch:03d} val_height={val.height_accuracy:.3f} val_transition={val.transition_accuracy:.3f} val_pos={val.position_rmse:.3f} val_num={val.number_rmse:.3f}"
            print(message, flush=True)
    model.load_state_dict(best["state"])
    metrics = {name: asdict(evaluate(model, loader, device)) for name, loader in loaders.items()}
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
        "evaluation_weights": "ema" if ema is not None else "online",
        "metrics": metrics,
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paths.ROOT / "data")
    parser.add_argument(
        "--annotation-source",
        choices=["curated", "automatic"],
        default="curated",
        help="Use curated ACMP stereogenic units or label-independent automatic ChiralFinder detections.",
    )
    parser.add_argument("--output", type=Path, default=paths.ROOT / "outputs" / "axial_ecd.json")
    parser.add_argument("--modes", nargs="+", default=["token", "gsf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Split each training batch for gradient accumulation; also bounds eval batches.",
    )
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument(
        "--n-chiral-heads",
        type=int,
        default=None,
        help="Number of attention heads receiving GSF-ChiRoPE corrections; default: half.",
    )
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gsf-scale", type=float, default=5.0)
    parser.add_argument(
        "--gsf-scope",
        choices=["global", "query_anchor", "incident_anchor"],
        default="global",
        help="Restrict rotary corrections to all, anchor-query, or anchor-incident pairs.",
    )
    parser.add_argument(
        "--gsf-support-mode",
        choices=["full", "random", "distance", "learned_topk"],
        default="full",
        help="Training-time correction support. Random and distance use fixed parity-even masks; learned_topk uses the learned pair gate.",
    )
    parser.add_argument(
        "--gsf-support-fraction",
        type=float,
        default=0.0,
        help="Sparse support density. Zero exactly matches the query-anchor entry count separately for every molecule and unit.",
    )
    parser.add_argument("--gsf-support-seed", type=int, default=20260828)
    parser.add_argument(
        "--gsf-rotary-mode",
        choices=["residual", "direct_replace"],
        default="residual",
        help="Use the residual rotary correction or directly replace selected heads' ordinary query-key score by the rotated compatibility.",
    )
    parser.add_argument(
        "--training-protocol",
        choices=["full", "pair_fraction", "one_side"],
        default="full",
        help="Use all paired labels, a target-blind fraction of complete training pairs, or one labelled configuration per training/validation pair.",
    )
    parser.add_argument("--training-pair-fraction", type=float, default=1.0)
    parser.add_argument("--training-protocol-seed", type=int, default=20260828)
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
        help="Use strict even/odd ECD readout or the original shared representation; parity_split gives Symbol a private even position-context head so its loss cannot update the supervised Position head.",
    )
    parser.add_argument(
        "--number-readout",
        choices=["categorical", "ordinal"],
        default="categorical",
        help="Predict peak count as unordered classes or ordered thresholds.",
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
    parser.add_argument(
        "--number-loss-weight",
        type=float,
        default=1.0,
        help="Weight of peak-number cross entropy in the ECD multi-task loss.",
    )
    parser.add_argument(
        "--number-ordinal-weight",
        type=float,
        default=0.0,
        help="Weight of expected peak-count MSE, an ordinal auxiliary loss.",
    )
    parser.add_argument(
        "--number-risk-weight",
        type=float,
        default=0.0,
        help="Weight of expected squared peak-count class distance.",
    )
    parser.add_argument(
        "--selection-number-weight",
        type=float,
        default=0.0,
        help="Validation checkpoint penalty applied to peak-number RMSE.",
    )
    parser.add_argument(
        "--position-loss-weight",
        type=float,
        default=1.0,
        help="Weight of categorical peak-position cross entropy.",
    )
    parser.add_argument(
        "--position-ordinal-weight",
        type=float,
        default=0.0,
        help="Weight of expected peak-position MSE.",
    )
    parser.add_argument(
        "--position-risk-weight",
        type=float,
        default=0.0,
        help="Weight of expected squared peak-position class distance.",
    )
    parser.add_argument(
        "--position-monotonic-weight",
        type=float,
        default=0.0,
        help="Weight of the expected-position strict-order margin loss.",
    )
    parser.add_argument(
        "--position-loss-reduction",
        choices=["peak", "molecule"],
        default="peak",
        help="Average Position cross entropy over peaks or equally over molecules.",
    )
    parser.add_argument(
        "--position-risk-reduction",
        choices=["peak_mse", "molecule_rmse"],
        default="peak_mse",
        help="Use the legacy peak MSE risk or the benchmark-aligned molecule RMSE surrogate.",
    )
    parser.add_argument(
        "--position-soft-label-sigma",
        type=float,
        default=0.0,
        help="Gaussian width for ordinal Position labels; zero keeps one-hot targets.",
    )
    parser.add_argument(
        "--selection-position-weight",
        type=float,
        default=0.0,
        help="Validation checkpoint penalty applied to peak-position RMSE.",
    )
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
    parser.add_argument("--log-every", type=int, default=5)
    parser.set_defaults(record_test_history=False)
    return parse_config(parser)


def main() -> None:
    args = parse_args()
    samples, split = load_ecd_samples(args.data_dir, annotation_source=args.annotation_source)
    audit = dataset_audit(samples)
    split, training_protocol_audit = apply_training_protocol(
        samples,
        split,
        protocol=args.training_protocol,
        pair_fraction=args.training_pair_fraction,
        protocol_seed=args.training_protocol_seed,
    )
    support_audit = None
    if args.gsf_support_mode in {"random", "distance"}:
        support_audit = controls.attach_gsf_support_masks(
            samples,
            mode=args.gsf_support_mode,
            fraction=args.gsf_support_fraction,
            support_seed=args.gsf_support_seed,
        )
        print("support_audit=" + json.dumps(support_audit), flush=True)
    print("training_protocol_audit=" + json.dumps(training_protocol_audit), flush=True)
    print("dataset_audit=" + json.dumps(audit), flush=True)
    results = [
        train_one(mode, seed, samples, split, args) for mode in args.modes for seed in args.seeds
    ]
    arg_dict = dict(vars(args))
    arg_dict["data_dir"] = str(arg_dict["data_dir"])
    arg_dict["output"] = str(arg_dict["output"])
    payload = {
        "selection_contract": f"maximize validation Symbol accuracy minus {args.selection_number_weight:g}*Number_RMSE minus {args.selection_position_weight:g}*Position_RMSE, with validation-loss tie-break; "
        + (
            "test metrics are retained per epoch for a retrospective diagnostic; "
            if args.record_test_history
            else "the test split is not evaluated during model selection; "
        )
        + "reported test metrics are evaluated after restoring the validation-selected checkpoint",
        "args": arg_dict,
        "dataset_audit": audit,
        "training_protocol_audit": training_protocol_audit,
        "support_audit": support_audit,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
