from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler

import gsf_chi.batching as batching
import gsf_chi.data.axial as data_axial
import gsf_chi.features as features
import gsf_chi.model as model
from gsf_chi.config import parse_config
from gsf_chi.paths import ROOT

CACHE_VERSION = "gsf-chi-moleculenet-v3"
TASKS = {
    "bbbp": {
        "file": "BBBP.csv",
        "smiles": "smiles",
        "targets": ["p_np"],
        "kind": "classification",
        "display": "BBBP",
    },
    "bace": {
        "file": "bace.csv",
        "smiles": "mol",
        "targets": ["Class"],
        "kind": "classification",
        "display": "BACE",
    },
    "clintox": {
        "file": "clintox.csv.gz",
        "smiles": "smiles",
        "targets": ["FDA_APPROVED", "CT_TOX"],
        "kind": "classification",
        "display": "ClinTox",
    },
    "sider": {
        "file": "sider.csv.gz",
        "smiles": "smiles",
        "targets": None,
        "kind": "classification",
        "display": "SIDER",
    },
    "freesolv": {
        "file": "SAMPL.csv",
        "smiles": "smiles",
        "targets": ["expt"],
        "kind": "regression",
        "display": "FreeSolv",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little")


def _embed_molecule(smiles: str) -> tuple[Chem.Mol, str]:
    """Build one deterministic 3D conformer, with a 2D fallback."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit parse failure")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if mol.GetNumAtoms() > 100:
        work = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(work)
        return (work, "2d_fallback")
    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = _stable_int(Chem.MolToSmiles(mol, canonical=True)) & 2147483647
    params.maxIterations = 100
    params.timeout = 3
    params.numThreads = 0
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    try:
        status = AllChem.EmbedMolecule(with_h, params)
    except (RuntimeError, ValueError):
        status = -1
    source = "etkdg"
    if status != 0:
        params.useRandomCoords = True
        try:
            status = AllChem.EmbedMolecule(with_h, params)
        except (RuntimeError, ValueError):
            status = -1
    if status == 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(with_h):
                AllChem.MMFFOptimizeMolecule(with_h, maxIters=50)
            else:
                AllChem.UFFOptimizeMolecule(with_h, maxIters=50)
        except (RuntimeError, ValueError):
            pass
        work = Chem.RemoveHs(with_h)
    else:
        work = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(work)
        source = "2d_fallback"
    if work.GetNumConformers() != 1:
        raise ValueError("conformer generation failure")
    return (work, source)


def _even_features(mol: Chem.Mol) -> tuple[Chem.Mol, np.ndarray]:
    even = Chem.Mol(mol)
    Chem.RemoveStereochemistry(even)
    node = features.atom_features(list(even.GetAtoms()), even, include_cip=False)
    node[:, -9:] = 0.0
    return (even, node)


def _center_shape(mol: Chem.Mol, center: int) -> tuple[float, float]:
    xyz = data_axial._coords(mol).astype(np.float64)
    neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors()]
    vectors = [xyz[index] - xyz[center] for index in neighbors]
    vectors = [vector / max(np.linalg.norm(vector), 1e-12) for vector in vectors]
    if len(vectors) >= 4:
        volume = abs(
            np.linalg.det(
                np.stack(
                    [vectors[0] - vectors[3], vectors[1] - vectors[3], vectors[2] - vectors[3]]
                )
            )
        )
    elif len(vectors) == 3:
        volume = abs(np.linalg.det(np.stack(vectors)))
    else:
        volume = 0.0
    mean_abs_dot = (
        float(
            np.mean(
                [abs(np.dot(vectors[i], vectors[j])) for i in range(len(vectors)) for j in range(i)]
            )
        )
        if len(vectors) > 1
        else 0.0
    )
    return (float(np.tanh(2.0 * volume)), mean_abs_dot)


def _sample_from_row(
    smiles: str, target: np.ndarray, target_mask: np.ndarray, row_index: int
) -> tuple[dict, dict]:
    stereo_mol, conformer_source = _embed_molecule(smiles)
    centers = [
        (int(index), label)
        for index, label in Chem.FindMolChiralCenters(
            stereo_mol, force=True, includeUnassigned=False, useLegacyImplementation=False
        )
        if label in {"R", "S"}
    ]
    even_mol, node = _even_features(stereo_mol)
    xyz = data_axial._coords(even_mol)
    graph_dist = np.asarray(Chem.GetDistanceMatrix(even_mol), dtype=np.float32)
    euclidean = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adjacency = np.asarray(rdmolops.GetAdjacencyMatrix(even_mol), dtype=np.float32)
    pair = np.stack(
        [np.clip(graph_dist, 0, 15) / 15.0, np.clip(euclidean, 0, 20) / 20.0, adjacency], axis=-1
    ).astype(np.float32)
    ranks = list(
        Chem.CanonicalRankAtoms(
            even_mol, breakTies=False, includeChirality=False, includeIsotopes=True
        )
    )
    cip_ranks = data_axial._cip_ranks(even_mol)
    relations, descriptors, phases, signs, confidence = ([], [], [], [], [])
    local_chi = np.zeros((even_mol.GetNumAtoms(), 1), dtype=np.float32)
    for center, label in centers:
        shape, mean_abs_dot = _center_shape(even_mol, center)
        unit = (center,)
        one_relation = data_axial._unit_relations(
            even_mol, unit, graph_dist, shape, mean_abs_dot, ranks=ranks, cip_ranks=cip_ranks
        )
        relations.append(one_relation)
        descriptors.append(
            data_axial._unit_descriptor(
                even_mol, unit, node, shape, mean_abs_dot, ranks, cip_ranks=cip_ranks
            )
        )
        phases.append(data_axial._phase_initialization(one_relation))
        sign = 1.0 if label == "R" else -1.0
        signs.append(sign)
        confidence.append(1.0)
        local_chi[center, 0] = sign
    n_atoms = even_mol.GetNumAtoms()
    if relations:
        unit_rel = np.stack(relations).astype(np.float32)
        unit_desc = np.stack(descriptors).astype(np.float32)
        phase_init = np.stack(phases).astype(np.float32)
    else:
        unit_rel = np.zeros((0, n_atoms, features.REL_DIM), dtype=np.float32)
        unit_desc = np.zeros((0, features.UNIT_DESC_DIM), dtype=np.float32)
        phase_init = np.zeros((0, n_atoms), dtype=np.float32)
    canonical = Chem.MolToSmiles(stereo_mol, canonical=True, isomericSmiles=True)
    sample = {
        "id": str(row_index),
        "base_id": row_index,
        "node": node,
        "pair": pair,
        "unit_rel": unit_rel,
        "unit_desc": unit_desc,
        "phase_init": phase_init,
        "chi": np.asarray(signs, dtype=np.float32),
        "rho": np.asarray(confidence, dtype=np.float32),
        "local_chi": local_chi,
        "label": 0,
        "target": target.astype(np.float32),
        "target_mask": target_mask.astype(bool),
        "smiles": canonical,
    }
    audit = {
        "atoms": n_atoms,
        "stereo_units": len(centers),
        "conformer_source": conformer_source,
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=stereo_mol, includeChirality=False),
    }
    return (sample, audit)


def _scaffold_split(samples: list[dict], scaffolds: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(index)
    ordered = sorted(groups.values(), key=lambda indices: (len(indices), indices[0]), reverse=True)
    n = len(samples)
    train_cutoff, val_cutoff = (0.8 * n, 0.9 * n)
    split = {"train": [], "validation": [], "test": []}
    for group in ordered:
        if len(split["train"]) + len(group) <= train_cutoff:
            split["train"].extend(group)
        elif len(split["train"]) + len(split["validation"]) + len(group) <= val_cutoff:
            split["validation"].extend(group)
        else:
            split["test"].extend(group)
    for values in split.values():
        values.sort()
    return split


def resplit_cache(cache_path: Path) -> dict:
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    samples = cached["samples"]
    scaffolds = []
    for sample in samples:
        mol = Chem.MolFromSmiles(sample["smiles"])
        if mol is None:
            raise ValueError(f"cached canonical SMILES is invalid: {sample['smiles']!r}")
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    cached["split"] = _scaffold_split(samples, scaffolds)
    cached["version"] = CACHE_VERSION
    cached["audit"]["split_sizes"] = {name: len(values) for name, values in cached["split"].items()}
    temporary = cache_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(cached, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)
    print(
        "resplit="
        + json.dumps({"task": cached["task"], "split_sizes": cached["audit"]["split_sizes"]}),
        flush=True,
    )
    return cached


def prepare_dataset(task: str, data_dir: Path, cache_dir: Path, force: bool) -> Path:
    spec = TASKS[task]
    source = data_dir / str(spec["file"])
    output = cache_dir / f"{task}.pkl"
    if output.exists() and (not force):
        with output.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("version") == CACHE_VERSION:
            print(f"cache hit task={task} samples={len(cached['samples'])}", flush=True)
            return output
        raise RuntimeError(f"incompatible cache: {output}; pass --force-prepare")
    frame = pd.read_csv(source)
    target_columns = spec["targets"]
    if target_columns is None:
        target_columns = [column for column in frame.columns if column != spec["smiles"]]
    samples, audits, failures = ([], [], [])
    started = time.monotonic()
    for row_index, row in frame.iterrows():
        raw_target = pd.to_numeric(row[list(target_columns)], errors="coerce").to_numpy(
            dtype=np.float32
        )
        mask = np.isfinite(raw_target)
        if not mask.any():
            failures.append({"row": int(row_index), "reason": "missing all targets"})
            continue
        raw_target = np.nan_to_num(raw_target, nan=0.0)
        try:
            sample, audit = _sample_from_row(
                str(row[spec["smiles"]]), raw_target, mask, int(row_index)
            )
        except Exception as error:
            failures.append({"row": int(row_index), "reason": f"{type(error).__name__}: {error}"})
            continue
        samples.append(sample)
        audits.append(audit)
        if len(samples) == 1 or len(samples) % 250 == 0:
            print(
                f"prepare task={task} usable={len(samples)} raw={row_index + 1}/{len(frame)}",
                flush=True,
            )
    split = _scaffold_split(samples, [item["scaffold"] for item in audits])
    split_sets = {name: set(values) for name, values in split.items()}
    if any((split_sets[a] & split_sets[b] for a in split_sets for b in split_sets if a < b)):
        raise AssertionError("split overlap")
    payload = {
        "version": CACHE_VERSION,
        "task": task,
        "kind": spec["kind"],
        "display": spec["display"],
        "target_columns": list(target_columns),
        "source": str(source.resolve()),
        "raw_rows": len(frame),
        "samples": samples,
        "split": split,
        "audit": {
            "usable_rows": len(samples),
            "excluded_rows": len(failures),
            "failures": failures,
            "split_sizes": {name: len(values) for name, values in split.items()},
            "scaffolds": len(set((item["scaffold"] for item in audits))),
            "stereogenic_molecules": int(sum((item["stereo_units"] > 0 for item in audits))),
            "stereo_units": int(sum((item["stereo_units"] for item in audits))),
            "two_dimensional_fallbacks": int(
                sum((item["conformer_source"] == "2d_fallback" for item in audits))
            ),
            "max_atoms": int(max((item["atoms"] for item in audits))),
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    (cache_dir / f"{task}.audit.json").write_text(
        json.dumps(payload["audit"], indent=2, ensure_ascii=False) + "\n"
    )
    print("prepared=" + json.dumps(payload["audit"], ensure_ascii=False), flush=True)
    return output


class SampleDataset(Dataset):
    def __init__(self, samples: list[dict], indices: Iterable[int]):
        self.samples = [samples[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class CostBatchSampler(Sampler[list[int]]):
    """Batch graphs under an all-unit/all-pair padding budget.

    A fixed molecule count is unsafe for SIDER: a few records contain roughly
    500 atoms and dozens of stereogenic units.  The budget estimates the
    dominant padded tensor as B * S_max * N_max^2, so ordinary small molecules
    retain the requested batch size while macromolecular outliers run alone.
    """

    def __init__(
        self, dataset: SampleDataset, max_batch_size: int, budget: int, shuffle: bool, seed: int
    ):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.budget = budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
            self.epoch += 1
        batch: list[int] = []
        max_atoms, max_units = (0, 1)
        for index in order:
            sample = self.dataset.samples[index]
            atoms = len(sample["node"])
            units = max(len(sample["chi"]), 1)
            proposed_atoms = max(max_atoms, atoms)
            proposed_units = max(max_units, units)
            proposed_size = len(batch) + 1
            proposed_cost = proposed_size * proposed_units * proposed_atoms**2
            if batch and (proposed_size > self.max_batch_size or proposed_cost > self.budget):
                yield batch
                batch = []
                max_atoms, max_units = (0, 1)
            batch.append(index)
            max_atoms = max(max_atoms, atoms)
            max_units = max(max_units, units)
        if batch:
            yield batch

    def __len__(self) -> int:
        return len(self.dataset)


def collate(samples: list[dict]) -> dict:
    if max((len(sample["chi"]) for sample in samples)) == 0:
        adjusted = []
        for sample in samples:
            one = dict(sample)
            n = len(sample["node"])
            one["unit_rel"] = np.zeros((1, n, features.REL_DIM), dtype=np.float32)
            one["unit_desc"] = np.zeros((1, features.UNIT_DESC_DIM), dtype=np.float32)
            one["phase_init"] = np.zeros((1, n), dtype=np.float32)
            one["chi"] = np.zeros(1, dtype=np.float32)
            one["rho"] = np.zeros(1, dtype=np.float32)
            adjusted.append(one)
        batch = batching.collate(adjusted)
        batch["unit_mask"].zero_()
    else:
        batch = batching.collate(samples)
    batch["target"] = torch.from_numpy(np.stack([x["target"] for x in samples]))
    batch["target_mask"] = torch.from_numpy(np.stack([x["target_mask"] for x in samples]))
    return batch


class PropertyModel(nn.Module):
    def __init__(self, output_dim: int, args: argparse.Namespace):
        super().__init__()
        self.encoder = model.MolecularModel(
            mode="gsf",
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            chiral_scale_init=args.gsf_scale,
            n_chiral_heads=args.n_chiral_heads,
            gsf_scope="global",
            condition_unit_axis=True,
            condition_unit_frequency=True,
            learn_pair_gate=True,
            use_phase_initialization=True,
            gsf_rotary_mode="residual",
        )
        self.encoder.head = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, args.d_model),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model, output_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.encoder.encode(batch))


@dataclass
class Evaluation:
    loss: float
    metric: float
    task_metrics: list[float]


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str,
    target_mean: float,
    target_std: float,
    amp: bool,
) -> Evaluation:
    model.eval()
    predictions, targets, masks = ([], [], [])
    losses, counts = (0.0, 0)
    for batch in loader:
        batch = _move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(batch)
            if kind == "classification":
                loss = _masked_bce(logits, batch["target"], batch["target_mask"])
            else:
                normalized = (batch["target"] - target_mean) / target_std
                loss = F.mse_loss(logits, normalized)
        size = len(batch["target"])
        losses += float(loss) * size
        counts += size
        if kind == "classification":
            predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        else:
            predictions.append((logits.float() * target_std + target_mean).cpu().numpy())
        targets.append(batch["target"].float().cpu().numpy())
        masks.append(batch["target_mask"].cpu().numpy())
    pred, target, mask = map(np.concatenate, (predictions, targets, masks))
    if kind == "classification":
        task_metrics = []
        for task_index in range(target.shape[1]):
            valid = mask[:, task_index]
            labels = target[valid, task_index]
            if valid.sum() == 0 or np.unique(labels).size < 2:
                task_metrics.append(float("nan"))
            else:
                task_metrics.append(float(roc_auc_score(labels, pred[valid, task_index])))
        metric = float(np.nanmean(task_metrics))
    else:
        valid = mask[:, 0]
        task_metrics = [float(math.sqrt(mean_squared_error(target[valid, 0], pred[valid, 0])))]
        metric = task_metrics[0]
    return Evaluation(loss=losses / max(counts, 1), metric=metric, task_metrics=task_metrics)


def train_one(
    task: str, seed: int, cache_path: Path, output_dir: Path, args: argparse.Namespace
) -> dict:
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    samples, split, kind = (cached["samples"], cached["split"], cached["kind"])
    set_seed(seed)
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    loaders = {}
    for name, shuffle in [("train", True), ("validation", False), ("test", False)]:
        dataset = SampleDataset(samples, split[name])
        batch_sampler = CostBatchSampler(
            dataset,
            max_batch_size=args.batch_size,
            budget=args.batch_budget,
            shuffle=shuffle,
            seed=seed,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
    train_targets = np.stack([samples[index]["target"] for index in split["train"]])
    train_masks = np.stack([samples[index]["target_mask"] for index in split["train"]])
    if kind == "regression":
        values = train_targets[train_masks].astype(np.float64)
        target_mean, target_std = (float(values.mean()), float(values.std()))
        target_std = max(target_std, 1e-08)
    else:
        target_mean, target_std = (0.0, 1.0)
    model = PropertyModel(len(cached["target_columns"]), args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr / 20
    )
    direction = 1.0 if kind == "classification" else -1.0
    best = {"score": -float("inf"), "epoch": 0, "state": None, "validation": None}
    epochs_without_gain = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
                if kind == "classification":
                    loss = _masked_bce(logits, batch["target"], batch["target_mask"])
                else:
                    normalized = (batch["target"] - target_mean) / target_std
                    loss = F.mse_loss(logits, normalized)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()
        validation = evaluate(
            model, loaders["validation"], device, kind, target_mean, target_std, amp
        )
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation.loss,
                "validation_metric": validation.metric,
            }
        )
        score = direction * validation.metric
        if np.isfinite(score) and score > best["score"] + args.min_delta:
            best = {
                "score": score,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "validation": validation,
            }
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"task={task} seed={seed} epoch={epoch:03d} val_metric={validation.metric:.6f} best_epoch={best['epoch']}",
                flush=True,
            )
        if epoch >= args.min_epochs and epochs_without_gain >= args.patience:
            print(f"early_stop task={task} seed={seed} epoch={epoch}", flush=True)
            break
    if best["state"] is None:
        raise RuntimeError("validation metric was never finite")
    model.load_state_dict(best["state"])
    test = evaluate(model, loaders["test"], device, kind, target_mean, target_std, amp)
    result = {
        "selection": "validation-only",
        "task": task,
        "kind": kind,
        "seed": seed,
        "best_epoch": best["epoch"],
        "validation": {
            "loss": best["validation"].loss,
            "metric": best["validation"].metric,
            "task_metrics": best["validation"].task_metrics,
        },
        "test": {"loss": test.loss, "metric": test.metric, "task_metrics": test.task_metrics},
        "target_normalization": {"mean": target_mean, "std": target_std},
        "parameters": sum((parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "config": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "n_chiral_heads": args.n_chiral_heads,
            "gsf_scale": args.gsf_scale,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "batch_budget": args.batch_budget,
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}.seed{seed}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        "result="
        + json.dumps({key: result[key] for key in ["task", "seed", "best_epoch", "test"]}),
        flush=True,
    )
    return result


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (float(array.mean()), float(array.std(ddof=1)))


def summarize(output_dir: Path, tex_path: Path, seeds: list[int]) -> dict:
    rows = {}
    for task in TASKS:
        results = []
        for seed in seeds:
            path = output_dir / f"{task}.seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            results.append(json.loads(path.read_text()))
        values = [float(result["test"]["metric"]) for result in results]
        mean, std = _mean_std(values)
        rows[task] = {"mean": mean, "std": std, "values": values}
    payload = {
        "selection": "validation-only checkpoint selection",
        "runs": len(seeds),
        "seeds": seeds,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    classification = [rows[name] for name in ["bbbp", "bace", "clintox", "sider"]]
    regression = rows["freesolv"]
    cells = [f"${100 * row['mean']:.1f}\\pm{100 * row['std']:.1f}$" for row in classification]
    cells.append(f"${regression['mean']:.3f}\\pm{regression['std']:.3f}$")
    tex = "% Machine-generated by gsf-chi moleculenet --summarize\n\\begin{table*}[t]\n  \\centering\n  \\caption{General molecular property prediction on MoleculeNet.\n  Classification entries are ROC-AUC (\\%); FreeSolv is\n  RMSE. GSF-$\\chi$ is trained from scratch without pretraining. We use a\n  fixed chirality-free scaffold split (80/10/10), select checkpoints on the\n  validation set, and report mean $\\pm$ sample standard deviation over three\n  runs.}\n  \\label{tab:moleculenet-generalization}\n  \\small\n  \\setlength{\\tabcolsep}{5pt}\n  \\begin{tabular}{lccccc}\n    \\toprule\n    Method & BBBP $\\uparrow$ & BACE $\\uparrow$ & ClinTox $\\uparrow$ & SIDER $\\uparrow$ & FreeSolv $\\downarrow$ \\\\\n    \\midrule\n    GSF-$\\chi$ (ours) & __GSF_CELLS__ \\\\\n    \\bottomrule\n  \\end{tabular}\n\\end{table*}\n".replace(
        "__GSF_CELLS__", " & ".join(cells)
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {tex_path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "moleculenet")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "outputs" / "gsf_moleculenet_3seeds" / "cache"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "gsf_moleculenet_3seeds" / "runs"
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=ROOT / "outputs" / "moleculenet_generalization.tex",
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resplit", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--batch-budget",
        type=int,
        default=1000000,
        help="Maximum padded B*S*N^2 cost; a single graph may exceed it.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-05)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-chiral-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gsf-scale", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parse_config(parser)


def main() -> None:
    args = parse_args()
    if not any([args.prepare, args.resplit, args.train, args.summarize]):
        raise SystemExit("choose at least one of --prepare, --resplit, --train, or --summarize")
    if args.prepare:
        for task in args.datasets:
            prepare_dataset(task, args.data_dir, args.cache_dir, force=args.force_prepare)
    if args.resplit:
        for task in args.datasets:
            resplit_cache(args.cache_dir / f"{task}.pkl")
    if args.train:
        if args.device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested but is not available")
        torch.set_float32_matmul_precision("high")
        for task in args.datasets:
            cache_path = args.cache_dir / f"{task}.pkl"
            for seed in args.seeds:
                train_one(task, seed, cache_path, args.output_dir, args)
    if args.summarize:
        summarize(args.output_dir, args.tex_path, args.seeds)
