"""Prepare a paired RotA conformer-robustness evaluation for ACMP test molecules."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

import gsf_chi.data.axial as gsf_data_axial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def base_id(sample_id: str) -> int:
    return abs(int(str(sample_id).split("_")[-1]))


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def isolate_conformer(mol: Chem.Mol, conformer: Chem.Conformer) -> Chem.Mol:
    output = Chem.Mol(mol)
    output.RemoveAllConformers()
    output.AddConformer(Chem.Conformer(conformer), assignId=True)
    return output


def reflect(mol: Chem.Mol) -> Chem.Mol:
    output = Chem.Mol(mol)
    conformer = output.GetConformer()
    xyz = np.asarray(conformer.GetPositions(), dtype=np.float64).copy()
    xyz[:, 2] *= -1.0
    for atom_index, position in enumerate(xyz):
        conformer.SetAtomPosition(atom_index, position)
    return output


def geometric_signs(mol: Chem.Mol, units: list[tuple[int, ...]], chiral_type: str) -> np.ndarray:
    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False, includeIsotopes=True)
    )
    cip_ranks = gsf_data_axial._cip_ranks(mol)
    return np.asarray(
        [
            gsf_data_axial._unit_chi_and_shape(
                mol, unit, {}, ranks, cip_ranks=cip_ranks, chiral_type=chiral_type
            )[0]
            for unit in units
        ],
        dtype=np.float32,
    )


def matched_determinant(result: dict, unit: tuple[int, ...]) -> float:
    return gsf_data_axial._matched_determinant_sign(result, unit)


def empty_finder_result() -> dict:
    return {
        "chiral axes": [],
        "quadrupole matrix": [],
        "determinant": [],
        "sign": [],
        "neighbor ids": [],
    }


def run_finder_with_idle_timeout(
    finder, n_cpus: int, timeout: float
) -> tuple[list[dict], list[int]]:
    """Collect native results until no worker returns within ``timeout`` seconds."""
    tasks = [(index, mol) for index, mol in enumerate(finder.mols)]
    results: list[dict | None] = [None] * len(tasks)
    pool = mp.Pool(processes=n_cpus)
    iterator = pool.imap_unordered(finder._process_one_mol_axial, tasks)
    completed = 0
    try:
        while completed < len(tasks):
            try:
                index, merged, _by_type = iterator.next(timeout=timeout)
            except mp.TimeoutError:
                break
            results[int(index)] = merged
            completed += 1
            if completed % 50 == 0:
                print(f"ChiralFinder {completed}/{len(tasks)}", flush=True)
    finally:
        if completed == len(tasks):
            pool.close()
        else:
            pool.terminate()
        pool.join()
    fallback = [index for index, result in enumerate(results) if result is None]
    return (
        [result if result is not None else empty_finder_result() for result in results],
        fallback,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--rota-pkl", type=Path, default=ROOT / "data/external/rota/RotA.pkl")
    parser.add_argument(
        "--chiralfinder-root", type=Path, default=ROOT / "data/external/chiralfinder"
    )
    parser.add_argument("--n-cpus", type=int, default=20)
    parser.add_argument("--finder-idle-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from chiralfinder import ChiralFinder

    with (args.data_dir / "hct_ecd_axial.pkl").open("rb") as handle:
        acmp_raw = pickle.load(handle)
    with (args.data_dir / "ecd_axial_index_split.pkl").open("rb") as handle:
        split = pickle.load(handle)
    with args.rota_pkl.open("rb") as handle:
        rota = pickle.load(handle)
    labels = pd.read_excel(args.data_dir / "axial_650.xlsx")
    optical_table = pd.read_csv(args.data_dir / "optical_rotation_589nm.csv")
    optical = {int(row.id): float(row.OR_589nm) for row in optical_table.itertuples(index=False)}
    rota_by_graph = {canonical_smiles(smiles): mol for smiles, mol in rota.items()}
    raw_by_id = {str(row["id"]): row for row in acmp_raw}
    test_bases = sorted({base_id(acmp_raw[int(index)]["id"]) for index in split["test_index"]})
    paired_rows: list[dict] = []
    excluded = []
    pair_index = 0
    orientation_counts = {"same": 0, "inverse": 0}
    for molecule_id in test_bases:
        label_row = labels.iloc[molecule_id]
        units = [tuple(map(int, unit)) for unit in ast.literal_eval(label_row["label"])]
        if any((len(unit) != 2 for unit in units)):
            excluded.append(
                {
                    "base_id": molecule_id,
                    "reason": "singleton stereogenic unit has no two-endpoint geometric gauge",
                }
            )
            continue
        reference_mol = raw_by_id[f"mol_{molecule_id}"]["rdkit_mol"]
        reference_sign = geometric_signs(reference_mol, units, str(label_row["chiral_type"]))
        if np.any(reference_sign == 0):
            excluded.append({"base_id": molecule_id, "reason": "zero reference sign"})
            continue
        rota_mol = rota_by_graph[canonical_smiles(str(label_row["SMILES"]))]
        for conformer_index, conformer in enumerate(rota_mol.GetConformers()):
            actual = isolate_conformer(rota_mol, conformer)
            current_sign = geometric_signs(actual, units, str(label_row["chiral_type"]))
            if np.array_equal(current_sign, reference_sign):
                base_mol = actual
                orientation = "same"
            elif np.array_equal(current_sign, -reference_sign):
                base_mol = reflect(actual)
                orientation = "inverse"
            else:
                excluded.append(
                    {
                        "base_id": molecule_id,
                        "conformer_index": conformer_index,
                        "reason": "partial or zero stereogenic-unit inversion",
                    }
                )
                continue
            orientation_counts[orientation] += 1
            mirror_mol = reflect(base_mol)
            for side, molecule, sample_id in (
                ("base", base_mol, f"mol_{molecule_id}"),
                ("mirror", mirror_mol, f"mol_-{molecule_id}"),
            ):
                template = copy.deepcopy(raw_by_id[sample_id])
                template["rdkit_mol"] = molecule
                template["id"] = sample_id
                template["original_base_id"] = molecule_id
                template["pair_index"] = pair_index
                template["conformer_index"] = conformer_index
                template["side"] = side
                template["eval_key"] = f"{molecule_id}:{conformer_index}:{side}"
                paired_rows.append(template)
            pair_index += 1
    finder = ChiralFinder([row["rdkit_mol"] for row in paired_rows], "molecules")
    finder_results, finder_fallback_indices = run_finder_with_idle_timeout(
        finder, args.n_cpus, args.finder_idle_timeout
    )
    if len(finder_results) != len(paired_rows):
        raise AssertionError("ChiralFinder output length mismatch")
    gsf_samples = []
    detection_hits = []
    determinant_inversions = []
    for row, result in zip(paired_rows, finder_results):
        molecule_id = int(row["original_base_id"])
        label_row = labels.iloc[molecule_id]
        units = [tuple(map(int, unit)) for unit in ast.literal_eval(label_row["label"])]
        detection_hits.extend(
            (
                any((set(unit) == set(predicted) for predicted in result["chiral axes"]))
                for unit in units
            )
        )
        sample = gsf_data_axial.build_sample(row, result, label_row, optical)
        sample.update(
            peak_num=int(row["peak_num"]),
            peak_position=np.asarray(row["peak_position"], dtype=np.int64),
            peak_height=np.asarray(row["peak_height"], dtype=np.int64),
            original_base_id=molecule_id,
            pair_index=int(row["pair_index"]),
            conformer_index=int(row["conformer_index"]),
            side=str(row["side"]),
            eval_key=str(row["eval_key"]),
        )
        sample["base_id"] = int(row["pair_index"])
        gsf_samples.append(sample)
    for start in range(0, len(paired_rows), 2):
        left_row, right_row = paired_rows[start : start + 2]
        left_result, right_result = finder_results[start : start + 2]
        units = [
            tuple(map(int, unit))
            for unit in ast.literal_eval(labels.iloc[int(left_row["original_base_id"])]["label"])
        ]
        left = np.asarray([matched_determinant(left_result, unit) for unit in units])
        right = np.asarray([matched_determinant(right_result, unit) for unit in units])
        if np.all(left != 0) and np.all(right != 0):
            determinant_inversions.append(bool(np.array_equal(left, -right)))
    parity_even_pairs = []
    chi_inverse_pairs = []
    for start in range(0, len(gsf_samples), 2):
        left, right = gsf_samples[start : start + 2]
        parity_even_pairs.append(
            np.allclose(left["node"], right["node"], atol=1e-06)
            and np.allclose(left["pair"], right["pair"], atol=1e-06)
            and np.allclose(left["unit_rel"], right["unit_rel"], atol=1e-06)
            and np.allclose(left["unit_desc"], right["unit_desc"], atol=1e-06)
        )
        chi_inverse_pairs.append(np.array_equal(left["chi"], -right["chi"]))
    audit = {
        "source": "ChiralFinder RotA",
        "source_molecules": len(rota),
        "source_conformers": sum((mol.GetNumConformers() for mol in rota.values())),
        "acmp_test_molecules": len(test_bases),
        "eligible_test_molecules": len({row["original_base_id"] for row in paired_rows}),
        "excluded": excluded,
        "physical_conformers": len(paired_rows) // 2,
        "paired_samples": len(paired_rows),
        "complete_enantiomer_pairs": len(paired_rows) // 2,
        "original_rota_orientation": orientation_counts,
        "curated_axis_detection_recall": float(np.mean(detection_hits)),
        "chiralfinder_native_samples": len(paired_rows) - len(finder_fallback_indices),
        "chiralfinder_coordinate_fallback_samples": len(finder_fallback_indices),
        "chiralfinder_coordinate_fallback_fraction": len(finder_fallback_indices)
        / len(paired_rows),
        "detected_pair_determinant_inversion_fraction": float(np.mean(determinant_inversions))
        if determinant_inversions
        else float("nan"),
        "all_gsf_parity_even_inputs_match": bool(all(parity_even_pairs)),
        "all_gsf_chi_vectors_invert": bool(all(chi_inverse_pairs)),
        "evaluation_contract": "locked ACMP checkpoints; ACMP test identities only; all published RotA conformers; physical coordinate reflection; no training or checkpoint selection",
    }
    payload = {
        "audit": audit,
        "rows": paired_rows,
        "finder_results": finder_results,
        "gsf_samples": gsf_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
