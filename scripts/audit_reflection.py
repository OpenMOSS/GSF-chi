"""Audit reflection consistency from coordinates through preprocessing and ECD output.

Unlike the algebraic property tests that negate the already constructed ``chi``
tensor, this audit reflects every RDKit conformer, transforms the geometric
stereochemical frames, recomputes their determinants, and then calls the public
``build_sample`` preprocessing entry point again.  It also records every place
where the chirality-free canonical gauge encounters an exact priority tie.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops

import gsf_chi.data.axial as gsf_data_axial
import gsf_chi.data.ecd as gsf_data_ecd
import gsf_chi.ecd as gsf_ecd
from gsf_chi.checkpoints import load_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFLECTION = np.diag([-1.0, 1.0, 1.0])
EVEN_FIELDS = ("node", "pair", "unit_rel", "unit_desc", "phase_init", "rho")


def reflected_molecule(molecule: Chem.Mol) -> Chem.Mol:
    mirrored = Chem.Mol(molecule)
    conformer = mirrored.GetConformer()
    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    reflected = positions @ REFLECTION.T
    for index, point in enumerate(reflected):
        conformer.SetAtomPosition(index, point.tolist())
    return mirrored


def reflected_stereo_record(record: dict) -> dict:
    """Transform geometric frames and recompute their pseudoscalar determinants."""
    mirrored = copy.deepcopy(record)
    transformed_frames = []
    determinants = []
    signs = []
    for wrapped_frame in record.get("quadrupole matrix", []):
        frames = np.asarray(wrapped_frame, dtype=np.float64).reshape(-1, 3, 3)
        reflected_frames = [frame @ REFLECTION.T for frame in frames]
        frame_determinants = [float(np.linalg.det(frame)) for frame in reflected_frames]
        transformed_frames.append(
            [reflected_frames[0]]
            if len(reflected_frames) == 1
            else [[frame] for frame in reflected_frames]
        )
        determinants.append(
            [frame_determinants[0]]
            if len(frame_determinants) == 1
            else [[value] for value in frame_determinants]
        )
        signs.append(
            [float(np.sign(frame_determinants[0]))]
            if len(frame_determinants) == 1
            else [[float(np.sign(value))] for value in frame_determinants]
        )
    mirrored["quadrupole matrix"] = transformed_frames
    mirrored["determinant"] = determinants
    mirrored["sign"] = signs
    return mirrored


def _priority_ties(keys: list[tuple]) -> int:
    counts = Counter(keys)
    return sum((count > 1 for count in counts.values()))


def canonicalization_audit(
    molecule: Chem.Mol, units: list[tuple[int, ...]]
) -> dict[str, int | float]:
    graph_ranks = list(
        Chem.CanonicalRankAtoms(
            molecule, breakTies=False, includeChirality=False, includeIsotopes=True
        )
    )
    unique_graph_ranks = list(
        Chem.CanonicalRankAtoms(
            molecule, breakTies=True, includeChirality=False, includeIsotopes=True
        )
    )
    cip_ranks = gsf_data_axial._cip_ranks(molecule)
    result: Counter = Counter()
    for unit in units:
        if len(unit) == 1:
            center = int(unit[0])
            neighbors = [atom.GetIdx() for atom in molecule.GetAtomWithIdx(center).GetNeighbors()]
            keys = [
                gsf_data_axial._priority_key(molecule, index, cip_ranks, graph_ranks)
                for index in neighbors
            ]
            result["singleton_units"] += 1
            result["singleton_neighbor_equivalence_classes"] += _priority_ties(keys)
            resolved_keys = [
                (*key, unique_graph_ranks[index]) for key, index in zip(keys, neighbors)
            ]
            result["singleton_unresolved_after_canonical_rank"] += _priority_ties(resolved_keys)
            continue
        if len(unit) != 2:
            result["unsupported_units"] += 1
            continue
        result["axial_units"] += 1
        endpoint_a, endpoint_b = map(int, unit)

        def candidates(endpoint: int, other: int) -> list[int]:
            path = rdmolops.GetShortestPath(molecule, endpoint, other)
            next_atom = int(path[1])
            return [
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(endpoint).GetNeighbors()
                if atom.GetIdx() != next_atom
            ]

        candidates_a = candidates(endpoint_a, endpoint_b)
        candidates_b = candidates(endpoint_b, endpoint_a)
        keys_a = [
            gsf_data_axial._priority_key(molecule, index, cip_ranks, graph_ranks)
            for index in candidates_a
        ]
        keys_b = [
            gsf_data_axial._priority_key(molecule, index, cip_ranks, graph_ranks)
            for index in candidates_b
        ]
        result["axial_substituent_equivalence_classes"] += _priority_ties(keys_a)
        result["axial_substituent_equivalence_classes"] += _priority_ties(keys_b)
        resolved_a = [(*key, unique_graph_ranks[index]) for key, index in zip(keys_a, candidates_a)]
        resolved_b = [(*key, unique_graph_ranks[index]) for key, index in zip(keys_b, candidates_b)]
        result["axial_substituent_unresolved_after_canonical_rank"] += _priority_ties(
            resolved_a
        ) + _priority_ties(resolved_b)

        def signature(endpoint: int, keys: list[tuple]) -> tuple:
            atom = molecule.GetAtomWithIdx(endpoint)
            return (
                tuple(sorted(keys, reverse=True)),
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                graph_ranks[endpoint],
            )

        if signature(endpoint_a, keys_a) == signature(endpoint_b, keys_b):
            result["axial_endpoint_equivalence_classes"] += 1
            result["axial_endpoint_resolved_by_canonical_rank"] += int(
                unique_graph_ranks[endpoint_a] != unique_graph_ranks[endpoint_b]
            )
            result["axial_endpoint_unresolved_after_canonical_rank"] += int(
                unique_graph_ranks[endpoint_a] == unique_graph_ranks[endpoint_b]
            )
        try:
            _, _, _, abs_sin, _ = gsf_data_axial._axis_vectors(
                molecule, (endpoint_a, endpoint_b), graph_ranks, cip_ranks=cip_ranks
            )
            result["axial_abs_sin_le_1e-6"] += int(abs_sin <= 1e-06)
            result["axial_abs_sin_le_1e-3"] += int(abs_sin <= 0.001)
        except ValueError:
            result["invalid_axial_gauges"] += 1
    return dict(result)


def maximum_array_residual(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0))


def flatten_numeric(value) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(flatten_numeric(item))
        return flattened
    return [float(value)]


def instantiate_model(run_json: Path, checkpoint: Path) -> gsf_ecd.ECDModel:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    return load_model(checkpoint, "axial_ecd", arguments=payload["args"])


def add_ecd_targets(sample: dict, raw_row: dict) -> dict:
    enriched = dict(sample)
    enriched["peak_num"] = int(raw_row["peak_num"])
    enriched["peak_position"] = np.asarray(
        raw_row["peak_position"][: gsf_ecd.N_PEAK_SLOTS], dtype=np.int64
    )
    enriched["peak_height"] = np.asarray(
        raw_row["peak_height"][: gsf_ecd.N_PEAK_SLOTS], dtype=np.int64
    )
    return enriched


def output_residuals(
    model: gsf_ecd.ECDModel,
    original_samples: list[dict],
    reflected_samples: list[dict],
    indices: list[int],
    batch_size: int,
) -> dict[str, float]:
    maxima = Counter()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            original = gsf_data_ecd.collate_ecd([original_samples[index] for index in chosen])
            reflected = gsf_data_ecd.collate_ecd([reflected_samples[index] for index in chosen])
            number, position, symbol, transition = model(original)
            r_number, r_position, r_symbol, r_transition = model(reflected)
            maxima["number_logit_max_abs"] = max(
                maxima["number_logit_max_abs"], float((number - r_number).abs().max())
            )
            maxima["position_logit_max_abs"] = max(
                maxima["position_logit_max_abs"], float((position - r_position).abs().max())
            )
            maxima["symbol_complement_logit_max_abs"] = max(
                maxima["symbol_complement_logit_max_abs"],
                float((symbol - r_symbol.flip(-1)).abs().max()),
            )
            if transition is not None and r_transition is not None:
                maxima["transition_logit_max_abs"] = max(
                    maxima["transition_logit_max_abs"],
                    float((transition - r_transition).abs().max()),
                )
    return dict(maxima)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.run_json is None) != (args.checkpoint is None):
        raise ValueError("--run-json and --checkpoint must be provided together")
    with (args.data_dir / "hct_ecd_axial.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    with (args.data_dir / "hct_ecd_axial_res.pkl").open("rb") as handle:
        stereo_records = pickle.load(handle)
    with (args.data_dir / "ecd_axial_index_split.pkl").open("rb") as handle:
        split = pickle.load(handle)
    labels = pd.read_excel(args.data_dir / "axial_650.xlsx")
    optical = pd.read_csv(args.data_dir / "optical_rotation_589nm.csv")
    optical_by_id = dict(zip(optical["id"].astype(int), optical["OR_589nm"].astype(float)))
    maxima = Counter()
    tie_totals = Counter()
    original_samples = []
    reflected_samples = []
    failed = []
    determinant_near_zero = Counter()
    for index, (row, record) in enumerate(zip(raw, stereo_records)):
        base_id, _ = gsf_data_axial._id_parts(row["id"])
        label_row = labels.iloc[base_id]
        units = [tuple(map(int, unit)) for unit in ast.literal_eval(label_row["label"])]
        molecule = row["rdkit_mol"]
        mirrored_row = dict(row)
        mirrored_row["rdkit_mol"] = reflected_molecule(molecule)
        mirrored_record = reflected_stereo_record(record)
        for value in flatten_numeric(record.get("determinant", [])):
            determinant_near_zero["abs_det_le_1e-8"] += int(abs(value) <= 1e-08)
            determinant_near_zero["abs_det_le_1e-4"] += int(abs(value) <= 0.0001)
        try:
            original = gsf_data_axial.build_sample(row, record, label_row, optical_by_id)
            mirrored = gsf_data_axial.build_sample(
                mirrored_row, mirrored_record, label_row, optical_by_id
            )
        except Exception as error:
            failed.append({"index": index, "id": row["id"], "error": repr(error)})
            continue
        for field in EVEN_FIELDS:
            maxima[f"{field}_max_abs"] = max(
                maxima[f"{field}_max_abs"], maximum_array_residual(original[field], mirrored[field])
            )
        maxima["chi_inversion_max_abs"] = max(
            maxima["chi_inversion_max_abs"],
            maximum_array_residual(original["chi"], -mirrored["chi"]),
        )
        maxima["local_chi_inversion_max_abs"] = max(
            maxima["local_chi_inversion_max_abs"],
            maximum_array_residual(original["local_chi"], -mirrored["local_chi"]),
        )
        original_cip = gsf_data_axial._cip_ranks(molecule)
        reflected_cip = gsf_data_axial._cip_ranks(mirrored_row["rdkit_mol"])
        maxima["cip_rank_max_abs"] = max(
            maxima["cip_rank_max_abs"], maximum_array_residual(original_cip, reflected_cip)
        )
        tie_totals.update(canonicalization_audit(molecule, units))
        original_samples.append(add_ecd_targets(original, row))
        reflected_samples.append(add_ecd_targets(mirrored, row))
    model_residual = None
    if args.run_json is not None:
        model = instantiate_model(args.run_json, args.checkpoint)
        if failed:
            raise RuntimeError("Cannot align model audit indices when preprocessing failed")
        if args.model_split == "all":
            indices = list(range(len(original_samples)))
        else:
            indices = [int(index) for index in split[f"{args.model_split}_index"]]
        model_residual = output_residuals(
            model, original_samples, reflected_samples, indices, args.batch_size
        )
    payload = {
        "contract": "RDKit coordinates are reflected by diag(-1,1,1); geometric stereochemical frames are reflected by the same operator and their determinants are recomputed before the standard build_sample pipeline.",
        "reflection_determinant": float(np.linalg.det(REFLECTION)),
        "molecules_audited": len(original_samples),
        "preprocessing_failures": failed,
        "preprocessing_residuals": dict(maxima),
        "canonicalization_counts": dict(tie_totals),
        "determinant_degeneracy_counts": dict(determinant_near_zero),
        "model_split": args.model_split if model_residual is not None else None,
        "model_output_residuals": model_residual,
        "run_json": str(args.run_json) if args.run_json else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
