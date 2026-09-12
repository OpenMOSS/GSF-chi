from __future__ import annotations

import bisect
import gc
import hashlib
import json
import math
import os
import pickle
import random
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
import gsf_chi.data.rs as data_rs

CACHE_FORMAT = "gsf-chirope-official-ranking-v1"
EXPECTED_SAMPLED_ROWS = {"train": 48384, "validation": 10368, "test": 10368}
REQUIRED_COLUMNS = {"ID", "SMILES_nostereo", "rdkit_mol_cistrans_stereo", "top_score"}
EXPECTED_PAIRS = {
    "stereo_group": {split: rows // 2 for split, rows in EXPECTED_SAMPLED_ROWS.items()},
    "legacy_chidek": {"train": 24192, "validation": 5183, "test": 5183},
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


def _selected_pair_rows(
    frame: pd.DataFrame,
    split: str,
    seed: int,
    max_pairs: int = 0,
    sampling_mode: str = "stereo_group",
) -> tuple[list[tuple[pd.Series, pd.Series]], int]:
    """Validate a split and reproduce the requested conformer-pair sampler."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{split} is missing required columns: {missing}")
    if not frame.index.is_unique:
        raise ValueError(f"{split} dataframe index must be unique")
    rng = random.Random(seed)
    pair_indices: list[tuple[object, object]] = []
    if sampling_mode == "legacy_chidek":
        groups: defaultdict[str, dict[str, list[object]]] = defaultdict(lambda: {"a": [], "b": []})
        for index, molecule_id in zip(frame.index, frame["ID"]):
            molecule_id = str(molecule_id)
            base_id = molecule_id.replace("@", "")
            side = "b" if "@@" in molecule_id else "a"
            groups[base_id][side].append(index)
        pair_indices = [
            (rng.choice(group["a"]), rng.choice(group["b"]))
            for group in groups.values()
            if group["a"] and group["b"]
        ]
        rng.shuffle(pair_indices)
    elif sampling_mode == "stereo_group":
        for base_key, group in frame.groupby("SMILES_nostereo", sort=False):
            stereo_ids = sorted(group["ID"].astype(str).unique())
            if len(stereo_ids) != 2:
                raise ValueError(
                    f"{split}/{base_key!r}: expected exactly two unique IDs, got {len(stereo_ids)}"
                )
            members: list[object] = []
            for stereo_id in stereo_ids:
                conformers = group[group["ID"].astype(str) == stereo_id]
                scores = conformers["top_score"].astype(float)
                if not np.isfinite(scores.to_numpy()).all():
                    raise ValueError(f"{split}/{stereo_id!r}: non-finite top_score")
                if not np.allclose(scores.to_numpy(), scores.iloc[0], rtol=0.0, atol=1e-08):
                    raise ValueError(f"{split}/{stereo_id!r}: top_score varies by conformer")
                members.append(conformers.index[rng.randrange(len(conformers))])
            pair_indices.append((members[0], members[1]))
    else:
        raise ValueError(f"unknown ranking sampling mode: {sampling_mode}")
    official_pairs = len(pair_indices)
    expected = EXPECTED_PAIRS[sampling_mode][split]
    if official_pairs != expected:
        raise ValueError(
            f"official ChIRo {split}/{sampling_mode} sampler produced {official_pairs} pairs; expected {expected}"
        )
    if max_pairs > 0:
        pair_indices = pair_indices[:max_pairs]
    selected = [(frame.loc[left], frame.loc[right]) for left, right in pair_indices]
    for left, right in selected:
        if str(left["SMILES_nostereo"]) != str(right["SMILES_nostereo"]):
            raise ValueError(
                f"{split}/{sampling_mode}: sampled pair has mismatched achiral scaffolds"
            )
        if not np.isfinite([float(left["top_score"]), float(right["top_score"])]).all():
            raise ValueError(f"{split}/{sampling_mode}: non-finite selected top_score")
    return (selected, official_pairs)


def build_sample(row: pd.Series, row_number: int) -> dict:
    """Build a chirality-clean graph sample with label-free geometric chi."""
    mol = row["rdkit_mol_cistrans_stereo"]
    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"row {row_number}: rdkit_mol_cistrans_stereo is not an RDKit Mol")
    if mol.GetNumConformers() != 1:
        raise ValueError(f"row {row_number}: expected one conformer, got {mol.GetNumConformers()}")
    if not mol.GetConformer().Is3D():
        raise ValueError(f"row {row_number}: conformer is not marked as 3D")
    center = data_rs._target_center(mol)
    chi_value, geometry_confidence = data_rs._geometry_chi(mol, center)
    even_mol, node = data_rs._even_atom_features(mol)
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
    base_key = str(row["SMILES_nostereo"])
    return {
        "id": str(row["ID"]),
        "base_key": base_key,
        "base_id": int.from_bytes(
            hashlib.blake2b(base_key.encode(), digest_size=8).digest(), "little"
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
        "label": float(row["top_score"]),
    }


def collate_pairs(pairs: list[tuple[dict, dict]]) -> dict:
    samples = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"expected two samples per pair, got {len(pair)}")
        left, right = pair
        if left["base_key"] != right["base_key"] or left["id"] == right["id"]:
            raise ValueError("invalid enantiomer pair in cached data")
        samples.extend((left, right))
    batch = batching.collate(samples)
    batch["label"] = torch.tensor([sample["label"] for sample in samples], dtype=torch.float32)
    batch["base_key"] = [sample["base_key"] for sample in samples]
    batch["stereo_id"] = [sample["id"] for sample in samples]
    return batch


def prepare_split(
    split: str,
    source: Path,
    cache_root: Path,
    pairs_per_shard: int,
    max_pairs: int,
    conformer_seed: int,
    log_every: int,
    sampling_mode: str = "stereo_group",
) -> dict:
    source = source.expanduser().resolve(strict=True)
    output_dir = cache_root / split
    index_path = output_dir / "index.json"
    fingerprint = _file_fingerprint(source)
    wanted_limit = max_pairs if max_pairs > 0 else None
    if index_path.exists():
        cached = json.loads(index_path.read_text())
        if (
            cached.get("format") == CACHE_FORMAT
            and cached.get("source") == fingerprint
            and (cached.get("pair_limit") == wanted_limit)
            and (cached.get("conformer_seed") == conformer_seed)
            and (cached.get("sampling_mode", "stereo_group") == sampling_mode)
            and all(((output_dir / item["file"]).is_file() for item in cached.get("shards", [])))
        ):
            print(f"cache hit: {split} ({cached['pairs']} pairs)", flush=True)
            return cached
        raise RuntimeError(
            f"stale/incompatible cache at {index_path}; use another --cache-dir or pass --rebuild-cache"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading trusted pickle: {source}", flush=True)
    frame = pd.read_pickle(source)
    selected, official_pairs = _selected_pair_rows(
        frame, split, conformer_seed, max_pairs=max_pairs, sampling_mode=sampling_mode
    )
    shards: list[dict] = []
    pending: list[tuple[dict, dict]] = []
    base_keys: set[str] = set()
    atom_count = 0
    start = time.monotonic()

    def flush() -> None:
        if not pending:
            return
        shard_name = f"shard-{len(shards):05d}.pkl"
        _atomic_pickle(output_dir / shard_name, list(pending))
        shards.append({"file": shard_name, "pairs": len(pending)})
        pending.clear()

    for pair_number, rows in enumerate(selected, start=1):
        try:
            pair = (
                build_sample(rows[0], 2 * (pair_number - 1)),
                build_sample(rows[1], 2 * (pair_number - 1) + 1),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to preprocess {split} pair {pair_number - 1}") from exc
        if pair[0]["base_key"] != pair[1]["base_key"]:
            raise RuntimeError(f"{split} pair {pair_number - 1} lost its base grouping")
        pending.append(pair)
        base_keys.add(pair[0]["base_key"])
        atom_count += len(pair[0]["node"]) + len(pair[1]["node"])
        if len(pending) >= pairs_per_shard:
            flush()
        if log_every > 0 and pair_number % log_every == 0:
            elapsed = max(time.monotonic() - start, 1e-06)
            print(
                f"prepare {split}: {pair_number}/{len(selected)} pairs ({pair_number / elapsed:.1f} pairs/s)",
                flush=True,
            )
    flush()
    metadata = {
        "format": CACHE_FORMAT,
        "split": split,
        "source": fingerprint,
        "conformer_seed": conformer_seed,
        "sampling_mode": sampling_mode,
        "pair_limit": wanted_limit,
        "official_pairs": official_pairs,
        "pairs": len(selected),
        "samples": 2 * len(selected),
        "base_keys": sorted(base_keys),
        "mean_atoms": atom_count / max(2 * len(selected), 1),
        "shards": shards,
    }
    _atomic_json(index_path, metadata)
    del selected, frame
    gc.collect()
    print(f"prepared {split}: {metadata['pairs']} pairs in {len(shards)} shards", flush=True)
    return metadata


class ShardedPairDataset(Dataset):
    """A map-style dataset whose indivisible item is one enantiomer pair."""

    def __init__(self, split_dir: Path, max_open_shards: int = 2):
        self.split_dir = split_dir
        self.metadata = json.loads((split_dir / "index.json").read_text())
        self.shards = self.metadata["shards"]
        self.offsets = [0]
        for shard in self.shards:
            self.offsets.append(self.offsets[-1] + int(shard["pairs"]))
        self.max_open_shards = max_open_shards
        self._loaded: OrderedDict[int, list[tuple[dict, dict]]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    def _load(self, shard_index: int) -> list[tuple[dict, dict]]:
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

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        return self._load(shard_index)[index - self.offsets[shard_index]]


class PairBatchSampler(Sampler[list[int]]):
    """Shuffle and batch pair indices without ever separating pair members."""

    def __init__(self, dataset: ShardedPairDataset, pairs_per_batch: int, seed: int, shuffle: bool):
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        self.dataset = dataset
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.pairs_per_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = []
            shard_order = torch.randperm(len(self.dataset.shards), generator=generator).tolist()
            for shard_index in shard_order:
                start, end = self.dataset.offsets[shard_index : shard_index + 2]
                local_order = torch.randperm(end - start, generator=generator).tolist()
                indices.extend((start + local for local in local_order))
        else:
            indices = list(range(len(self.dataset)))
        for start in range(0, len(indices), self.pairs_per_batch):
            yield indices[start : start + self.pairs_per_batch]
