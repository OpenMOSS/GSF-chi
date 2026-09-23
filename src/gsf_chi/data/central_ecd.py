from __future__ import annotations

import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

import gsf_chi.data.axial as data_axial
import gsf_chi.features as features

N_PEAK_SLOTS = 9
N_POSITION_CLASSES = 20
SPECTRUM_TARGET_VERSION = "normalized-abs-20-v1"


def _molecule(entry: dict) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(entry["smiles"]))
    if mol is None:
        raise ValueError(f"RDKit could not parse {entry['smiles']!r}")
    positions = np.asarray(entry["info"]["atom_pos"], dtype=np.float64)
    if mol.GetNumAtoms() != len(positions):
        raise ValueError(
            f"Atom/coordinate mismatch for id={entry['id']}: {mol.GetNumAtoms()} != {len(positions)}"
        )
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_index, (x, y, z) in enumerate(positions):
        conformer.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    return mol


def _central_unit(mol: Chem.Mol) -> tuple[int, float, float, float]:
    centers = [
        (int(atom_index), label)
        for atom_index, label in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
        if label in {"R", "S"}
    ]
    if len(centers) != 1:
        raise ValueError(f"Expected one assigned tetrahedral center, found {centers}")
    center, label = centers[0]
    chi = 1.0 if label == "R" else -1.0
    xyz = data_axial._coords(mol).astype(np.float64)
    neighbours = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()]
    vectors = [xyz[index] - xyz[center] for index in neighbours]
    vectors = [vector / max(np.linalg.norm(vector), 1e-12) for vector in vectors]
    if len(vectors) >= 4:
        volume = abs(
            np.linalg.det(
                np.stack(
                    [vectors[0] - vectors[3], vectors[1] - vectors[3], vectors[2] - vectors[3]],
                    axis=0,
                )
            )
        )
    elif len(vectors) == 3:
        volume = abs(np.linalg.det(np.stack(vectors, axis=0)))
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
    return (center, chi, float(np.tanh(2.0 * volume)), mean_abs_dot)


def _graph_sample(entry: dict, base_id: int, sample_id: str) -> dict:
    mol = _molecule(entry)
    center, chi, shape, mean_abs_dot = _central_unit(mol)
    xyz = data_axial._coords(mol)
    graph_dist = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float32)
    euclidean = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adjacency = np.asarray(rdmolops.GetAdjacencyMatrix(mol), dtype=np.float32)
    pair = np.stack(
        [np.clip(graph_dist, 0, 15) / 15.0, np.clip(euclidean, 0, 20) / 20.0, adjacency], axis=-1
    ).astype(np.float32)
    node = features.atom_features(list(mol.GetAtoms()), mol, include_cip=False)
    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False, includeIsotopes=True)
    )
    cip_ranks = data_axial._cip_ranks(mol)
    unit = (center,)
    relations = data_axial._unit_relations(
        mol, unit, graph_dist, shape, mean_abs_dot, ranks=ranks, cip_ranks=cip_ranks
    )
    descriptor = data_axial._unit_descriptor(
        mol, unit, node, shape, mean_abs_dot, ranks, cip_ranks=cip_ranks
    )
    local_chi = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
    local_chi[center, 0] = chi
    sample = {
        "id": sample_id,
        "base_id": int(base_id),
        "node": node,
        "pair": pair,
        "unit_rel": relations[None],
        "unit_desc": descriptor[None],
        "phase_init": data_axial._phase_initialization(relations)[None],
        "chi": np.asarray([chi], dtype=np.float32),
        "rho": np.asarray([1.0], dtype=np.float32),
        "local_chi": local_chi,
        "label": 0,
    }
    return sample


def _spectrum_labels(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    raw = table["ECD (Mdeg)"].to_numpy()
    signal = np.asarray([int(x) if x > 1 or x < -1 else 0 for x in raw], dtype=float)
    nonzero = np.flatnonzero(signal)
    if len(nonzero):
        signal = signal[nonzero[0] : nonzero[-1] + 1]
    distance = max(int(len(signal) / (N_POSITION_CLASSES - 1)), 1)
    sequence = signal[::distance][:N_POSITION_CLASSES].tolist()
    if len(sequence) < N_POSITION_CLASSES:
        sequence.extend([0.0] * (N_POSITION_CLASSES - len(sequence)))
    positive_max = max(max(sequence), 1.0)
    negative_min = min(min(sequence), -1.0)
    normalized = [
        value * 100.0 / positive_max if value >= 0 else value * -100.0 / negative_min
        for value in sequence
    ]
    peak_positions = []
    for index in range(1, len(normalized) - 1):
        if (
            normalized[index - 1] < normalized[index] > normalized[index + 1]
            or normalized[index - 1] > normalized[index] < normalized[index + 1]
        ):
            peak_positions.append(index)
    if len(peak_positions) >= N_PEAK_SLOTS:
        raise ValueError(f"{path} has {len(peak_positions)} peaks")
    peak_heights = [int(normalized[index] >= 0) for index in peak_positions]
    positions = np.full(N_PEAK_SLOTS, -1, dtype=np.int64)
    heights = np.full(N_PEAK_SLOTS, -1, dtype=np.int64)
    positions[: len(peak_positions)] = peak_positions
    heights[: len(peak_heights)] = peak_heights
    spectrum_abs = np.abs(np.asarray(normalized, dtype=np.float32)) / 100.0
    return (len(peak_positions), positions, heights, spectrum_abs)


def _spectrum_paths(raw_root: Path) -> dict[int, Path]:
    paths = {}
    for path in raw_root.glob("*ECD/data/*.csv"):
        spectrum_id = int(path.stem)
        if spectrum_id in paths:
            raise ValueError(f"Duplicate spectrum id {spectrum_id}")
        paths[spectrum_id] = path
    return paths


def build_dataset(raw_root: Path, graph_path: Path) -> tuple[list[dict], dict, dict]:
    records = np.load(graph_path, allow_pickle=True).tolist()
    by_hand: dict[int, list[int]] = defaultdict(list)
    for row_index, entry in enumerate(records):
        by_hand[int(entry["hand_id"])].append(row_index)
    spectra = _spectrum_paths(raw_root)
    pairs: list[tuple[dict, dict]] = []
    excluded = Counter()
    for spectrum_id, spectrum_path in sorted(spectra.items()):
        row_index = spectrum_id - 1
        entry = records[row_index]
        partners = by_hand[int(entry["hand_id"])]
        if len(partners) != 2:
            excluded["incomplete_hand_group"] += 1
            continue
        other_index = partners[0] if partners[1] == row_index else partners[1]
        try:
            first = _graph_sample(entry, spectrum_id, f"{spectrum_id}_observed")
            second = _graph_sample(records[other_index], spectrum_id, f"{spectrum_id}_opposite")
        except ValueError as error:
            if "Expected one assigned tetrahedral center" in str(error):
                excluded["no_single_tetrahedral_center"] += 1
                continue
            raise
        peak_num, peak_position, peak_height, spectrum_abs = _spectrum_labels(spectrum_path)
        first.update(
            peak_num=peak_num,
            peak_position=peak_position,
            peak_height=peak_height,
            spectrum_abs=spectrum_abs,
            spectrum_target_version=SPECTRUM_TARGET_VERSION,
        )
        opposite_height = peak_height.copy()
        opposite_height[:peak_num] = 1 - opposite_height[:peak_num]
        second.update(
            peak_num=peak_num,
            peak_position=peak_position.copy(),
            peak_height=opposite_height,
            spectrum_abs=spectrum_abs.copy(),
            spectrum_target_version=SPECTRUM_TARGET_VERSION,
        )
        pairs.append((first, second))
    pair_order = list(range(len(pairs)))
    random.Random(42).shuffle(pair_order)
    train_end = int(0.8 * len(pair_order))
    val_end = int(0.9 * len(pair_order))
    pair_splits = {
        "train_index": pair_order[:train_end],
        "val_index": pair_order[train_end:val_end],
        "test_index": pair_order[val_end:],
    }
    samples = [sample for pair in pairs for sample in pair]
    split = {
        name: [
            sample_index
            for pair_index in indices
            for sample_index in (2 * pair_index, 2 * pair_index + 1)
        ]
        for name, indices in pair_splits.items()
    }
    audit = {
        "public_spectra": len(spectra),
        "central_pairs": len(pairs),
        "molecules": len(samples),
        "excluded": dict(excluded),
        "spectrum_target_version": SPECTRUM_TARGET_VERSION,
        "split_molecules": {name: len(indices) for name, indices in split.items()},
        "all_pair_labels_complement": all(
            (
                np.array_equal(a["peak_position"], b["peak_position"])
                and a["peak_num"] == b["peak_num"]
                and np.all(
                    a["peak_height"][: a["peak_num"]] + b["peak_height"][: b["peak_num"]] == 1
                )
                for a, b in pairs
            )
        ),
        "all_pair_chi_inverts": all((np.array_equal(a["chi"], -b["chi"]) for a, b in pairs)),
    }
    return (samples, split, audit)


def load_dataset(args):
    if args.cache.exists() and (not args.rebuild_cache):
        with args.cache.open("rb") as handle:
            payload = pickle.load(handle)
        samples = payload["samples"]
        if not samples or samples[0].get("spectrum_target_version") != SPECTRUM_TARGET_VERSION:
            raise ValueError(
                "Central ECD cache predates the current peak targets; rebuild with --rebuild-cache."
            )
        return (samples, payload["split"], payload["audit"])
    samples, split, audit = build_dataset(args.raw_root, args.graph_path)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as handle:
        pickle.dump(
            {"samples": samples, "split": split, "audit": audit},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return (samples, split, audit)
