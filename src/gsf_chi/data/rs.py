from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset, Sampler

import gsf_chi.batching as batching
import gsf_chi.data.axial as data_axial
import gsf_chi.features as features

CACHE_FORMAT = "gsf-chirope-official-rs-v1"
REQUIRED_COLUMNS = {
    "ID",
    "SMILES_nostereo",
    "rdkit_mol_cistrans_stereo",
    "RS_label",
    "RS_label_binary",
}


def _file_fingerprint(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    paths = {"train": args.train_pkl, "validation": args.val_pkl, "test": args.test_pkl}
    if args.data_manifest is None:
        return {key: path.expanduser() for key, path in paths.items()}
    manifest_path = args.data_manifest.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("data manifest must have schema_version=1 and a files object")
    logical = {"train": "RS_train", "validation": "RS_validation", "test": "RS_test"}
    resolved = {}
    for split, key in logical.items():
        relative = manifest["files"].get(key)
        if relative is None:
            resolved[split] = paths[split].expanduser()
            continue
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(f"manifest path for {key} must be relative: {relative}")
        resolved[split] = manifest_path.parent / candidate
    return resolved


def _target_center(mol: Chem.Mol) -> int:
    """Locate the annotated stereo atom without consuming its R/S value.

    ChIRo has one assigned target center.  A few structures contain additional
    RDKit ``?`` candidates; those are not the benchmark target.  We use only the
    atom index of the assigned tag and never its CIP value.
    """
    centers = Chem.FindMolChiralCenters(
        mol, force=True, includeUnassigned=True, useLegacyImplementation=False
    )
    assigned = [int(index) for index, cip in centers if cip in {"R", "S"}]
    if len(assigned) != 1:
        raise ValueError(f"expected exactly one assigned R/S center; centers={centers!r}")
    return assigned[0]


def _even_atom_features(mol: Chem.Mol) -> tuple[Chem.Mol, np.ndarray]:
    """Return a topology-equivalent molecule and atom features without stereo tags."""
    even_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(even_mol)
    node = features.atom_features(list(even_mol.GetAtoms()), even_mol, include_cip=False)
    node[:, -9:] = 0.0
    return (even_mol, node)


def _geometry_chi(mol: Chem.Mol, center: int) -> tuple[float, float]:
    """Compute a label-free central determinant sign and normalized magnitude.

    Hydrogens are added *after* removing stereochemical tags.  Their coordinates
    are consequently reconstructed from the supplied 3D conformer rather than
    from an R/S annotation.  Three-coordinate pyramidal centers use the opposite
    neighbor centroid as a lone-pair pseudo-site, matching ChiDeK's fallback
    construction.  Sites are ordered using only chirality-free graph ranks and
    elemental invariants.
    """
    work = Chem.Mol(mol)
    Chem.RemoveStereochemistry(work)
    work = Chem.AddHs(work, addCoords=True)
    if work.GetNumConformers() != 1:
        raise ValueError("could not construct a single hydrogen-completed conformer")
    neighbors = [atom.GetIdx() for atom in work.GetAtomWithIdx(center).GetNeighbors()]
    if len(neighbors) not in {3, 4}:
        raise ValueError(
            f"target center {center} must have three or four substituent sites; got {len(neighbors)} after AddHs"
        )
    graph_ranks = list(
        Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False, includeIsotopes=True)
    )
    cip_ranks = data_axial._cip_ranks(work)
    neighbors.sort(
        key=lambda index: data_axial._priority_key(work, index, cip_ranks, graph_ranks),
        reverse=True,
    )
    xyz = data_axial._coords(work).astype(np.float64)
    center_xyz = xyz[center]
    vectors = np.stack([xyz[index] - center_xyz for index in neighbors])
    if len(neighbors) == 3:
        pseudo = -vectors.mean(axis=0, keepdims=True)
        vectors = np.concatenate([vectors, pseudo], axis=0)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    determinant = float(
        np.linalg.det(
            np.stack(
                [vectors[0] - vectors[3], vectors[1] - vectors[3], vectors[2] - vectors[3]], axis=0
            )
        )
    )
    if abs(determinant) < 1e-08:
        raise ValueError(f"degenerate tetrahedral geometry at center {center}")
    return (float(np.sign(determinant)), float(np.clip(abs(determinant), 0.0, 2.0) / 2.0))


def build_sample(row: pd.Series, row_number: int) -> dict:
    mol = row["rdkit_mol_cistrans_stereo"]
    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"row {row_number}: rdkit_mol_cistrans_stereo is not an RDKit Mol")
    if mol.GetNumConformers() != 1:
        raise ValueError(f"row {row_number}: expected one conformer, got {mol.GetNumConformers()}")
    if not mol.GetConformer().Is3D():
        raise ValueError(f"row {row_number}: conformer is not marked as 3D")
    label_name = str(row["RS_label"])
    label = int(row["RS_label_binary"])
    expected_binary = {"R": 0, "S": 1}.get(label_name)
    if expected_binary is None or label != expected_binary:
        raise ValueError(
            f"row {row_number}: inconsistent RS_label/RS_label_binary: {label_name!r}/{label!r}"
        )
    center = _target_center(mol)
    chi_value, geometry_confidence = _geometry_chi(mol, center)
    even_mol, node = _even_atom_features(mol)
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
    unit = (center,)
    relations = data_axial._unit_relations(
        even_mol,
        unit,
        graph_dist,
        abs_sin=geometry_confidence,
        cos_phi=0.0,
        ranks=ranks,
        cip_ranks=cip_ranks,
    )
    descriptor = data_axial._unit_descriptor(
        even_mol,
        unit,
        node,
        abs_sin=geometry_confidence,
        cos_phi=0.0,
        ranks=ranks,
        cip_ranks=cip_ranks,
    )
    local_chi = np.zeros((even_mol.GetNumAtoms(), 1), dtype=np.float32)
    local_chi[center, 0] = chi_value
    return {
        "id": str(row["ID"]),
        "base_key": str(row["SMILES_nostereo"]),
        "base_id": int.from_bytes(
            hashlib.blake2b(str(row["SMILES_nostereo"]).encode(), digest_size=8).digest(), "little"
        )
        % (2**63 - 1),
        "node": node,
        "pair": pair,
        "unit_rel": relations[None].astype(np.float32),
        "unit_desc": descriptor[None].astype(np.float32),
        "phase_init": data_axial._phase_initialization(relations)[None],
        "chi": np.asarray([chi_value], dtype=np.float32),
        "rho": np.asarray([geometry_confidence], dtype=np.float32),
        "local_chi": local_chi,
        "label": label,
    }


def collate(samples: list[dict]) -> dict:
    batch = batching.collate(samples)
    batch["base_key"] = [sample["base_key"] for sample in samples]
    batch["stereo_id"] = [sample["id"] for sample in samples]
    return batch


def prepare_split(
    split: str, source: Path, cache_root: Path, shard_size: int, max_samples: int, log_every: int
) -> dict:
    source = source.expanduser().resolve(strict=True)
    output_dir = cache_root / split
    index_path = output_dir / "index.json"
    fingerprint = _file_fingerprint(source)
    wanted_limit = max_samples if max_samples > 0 else None
    if index_path.exists():
        cached = json.loads(index_path.read_text())
        if (
            cached.get("format") == CACHE_FORMAT
            and cached.get("source") == fingerprint
            and (cached.get("sample_limit") == wanted_limit)
            and all(((output_dir / item["file"]).is_file() for item in cached.get("shards", [])))
        ):
            print(f"cache hit: {split} ({cached['samples']} samples)", flush=True)
            return cached
        raise RuntimeError(
            f"stale/incompatible cache at {index_path}; use a different --cache-dir or pass --rebuild-cache"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading trusted pickle: {source}", flush=True)
    frame = pd.read_pickle(source)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.iloc[:max_samples]
    shards, pending = ([], [])
    base_keys: set[str] = set()
    labels = defaultdict(int)
    atom_count = 0
    start = time.monotonic()

    def flush() -> None:
        if not pending:
            return
        shard_name = f"shard-{len(shards):05d}.pkl"
        _atomic_pickle(output_dir / shard_name, list(pending))
        shards.append({"file": shard_name, "samples": len(pending)})
        pending.clear()

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        try:
            sample = build_sample(row, position - 1)
        except Exception as exc:
            raise RuntimeError(f"failed to preprocess {split} row {position - 1}") from exc
        pending.append(sample)
        base_keys.add(sample["base_key"])
        labels[str(sample["label"])] += 1
        atom_count += len(sample["node"])
        if len(pending) >= shard_size:
            flush()
        if log_every > 0 and position % log_every == 0:
            elapsed = max(time.monotonic() - start, 1e-06)
            print(
                f"prepare {split}: {position}/{len(frame)} ({position / elapsed:.1f} conformers/s)",
                flush=True,
            )
    flush()
    metadata = {
        "format": CACHE_FORMAT,
        "split": split,
        "source": fingerprint,
        "sample_limit": wanted_limit,
        "samples": len(frame),
        "stereoisomers": int(frame["ID"].nunique()),
        "base_molecules": len(base_keys),
        "base_keys": sorted(base_keys),
        "label_counts": dict(labels),
        "mean_atoms": atom_count / max(len(frame), 1),
        "shards": shards,
    }
    _atomic_json(index_path, metadata)
    del frame
    gc.collect()
    print(f"prepared {split}: {metadata['samples']} samples in {len(shards)} shards", flush=True)
    return metadata


class ShardedDataset(Dataset):
    def __init__(self, split_dir: Path, max_open_shards: int = 2):
        self.split_dir = split_dir
        self.metadata = json.loads((split_dir / "index.json").read_text())
        self.shards = self.metadata["shards"]
        self.offsets = [0]
        for shard in self.shards:
            self.offsets.append(self.offsets[-1] + int(shard["samples"]))
        self.max_open_shards = max_open_shards
        self._loaded: OrderedDict[int, list[dict]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    def _load(self, shard_index: int) -> list[dict]:
        if shard_index in self._loaded:
            result = self._loaded.pop(shard_index)
            self._loaded[shard_index] = result
            return result
        with (self.split_dir / self.shards[shard_index]["file"]).open("rb") as handle:
            result = pickle.load(handle)
        self._loaded[shard_index] = result
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return result

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        return self._load(shard_index)[index - self.offsets[shard_index]]


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shards and rows while retaining mostly sequential shard access."""

    def __init__(self, dataset: ShardedDataset, seed: int):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.dataset.shards), generator=generator).tolist()
        for shard_index in shard_order:
            start, end = self.dataset.offsets[shard_index : shard_index + 2]
            local_order = torch.randperm(end - start, generator=generator).tolist()
            yield from (start + local for local in local_order)
