from __future__ import annotations

import ast
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

from gsf_chi.features import NODE_FEATURE_DIM, REL_DIM, UNIT_DESC_DIM, atom_features


def _read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def _id_parts(text: str) -> tuple[int, bool]:
    suffix = text.split("_")[-1]
    return (abs(int(suffix)), "-" in suffix)


def _coords(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)


def _cip_ranks(mol: Chem.Mol) -> list[int]:
    """Return RDKit CIP ranks, falling back to chirality-free graph ranks."""
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if not all((atom.HasProp("_CIPRank") for atom in work.GetAtoms())):
        try:
            from rdkit.Chem import rdCIPLabeler

            rdCIPLabeler.AssignCIPLabels(work)
        except (ImportError, RuntimeError):
            pass
    fallback = list(
        Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False, includeIsotopes=True)
    )
    return [
        int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else fallback[i]
        for i, atom in enumerate(work.GetAtoms())
    ]


def _priority_key(
    mol: Chem.Mol, atom_index: int, cip_ranks: list[int], graph_ranks: list[int]
) -> tuple[int, int, int, int, int]:
    atom = mol.GetAtomWithIdx(atom_index)
    return (
        cip_ranks[atom_index],
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        int(atom.GetFormalCharge()),
        graph_ranks[atom_index],
    )


def _axis_roles(
    mol: Chem.Mol, axis: tuple[int, int], graph_ranks: list[int], cip_ranks: list[int] | None = None
) -> dict:
    """Canonicalize A/B and return CIP-prioritized substituent roles.

    The canonical direction is derived only from parity-even graph/CIP invariants,
    so reversing the endpoints in an input record does not change the descriptor.
    """
    original_a, original_b = map(int, axis)
    cip_ranks = _cip_ranks(mol) if cip_ranks is None else cip_ranks

    def one_side(endpoint: int, other: int) -> tuple[int, list[int]]:
        path = rdmolops.GetShortestPath(mol, endpoint, other)
        if len(path) < 2:
            raise ValueError(f"Invalid axial endpoints {axis}")
        next_atom = int(path[1])
        candidates = [
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(endpoint).GetNeighbors()
            if atom.GetIdx() != next_atom
        ]
        candidates.sort(
            key=lambda idx: _priority_key(mol, idx, cip_ranks, graph_ranks), reverse=True
        )
        return (next_atom, candidates)

    next_a0, cand_a0 = one_side(original_a, original_b)
    next_b0, cand_b0 = one_side(original_b, original_a)

    def endpoint_signature(endpoint: int, candidates: list[int]) -> tuple:
        atom = mol.GetAtomWithIdx(endpoint)
        substituent_signature = tuple(
            (_priority_key(mol, idx, cip_ranks, graph_ranks) for idx in candidates)
        )
        return (
            substituent_signature,
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            graph_ranks[endpoint],
        )

    signature_a = endpoint_signature(original_a, cand_a0)
    signature_b = endpoint_signature(original_b, cand_b0)
    if signature_b > signature_a:
        a, b = (original_b, original_a)
    elif signature_a > signature_b:
        a, b = (original_a, original_b)
    else:
        unique_ranks = list(
            Chem.CanonicalRankAtoms(
                mol, breakTies=True, includeChirality=False, includeIsotopes=True
            )
        )
        a, b = (
            (original_a, original_b)
            if unique_ranks[original_a] <= unique_ranks[original_b]
            else (original_b, original_a)
        )
    next_a, cand_a = one_side(a, b)
    next_b, cand_b = one_side(b, a)
    if not cand_a or not cand_b:
        raise ValueError(f"Axis {axis} has no external substituent on one side")
    return {
        "a": a,
        "b": b,
        "next_a": next_a,
        "next_b": next_b,
        "cand_a": cand_a,
        "cand_b": cand_b,
        "high_a": cand_a[0],
        "low_a": cand_a[1] if len(cand_a) > 1 else None,
        "high_b": cand_b[0],
        "low_b": cand_b[1] if len(cand_b) > 1 else None,
        "cip_ranks": cip_ranks,
    }


def _axis_vectors(
    mol: Chem.Mol, axis: tuple[int, int], ranks: list[int], cip_ranks: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the canonical directed axis and CIP-high projected vectors."""
    roles = _axis_roles(mol, axis, ranks, cip_ranks=cip_ranks)
    a, b = (roles["a"], roles["b"])
    xyz = _coords(mol).astype(np.float64)
    z = xyz[b] - xyz[a]
    z /= max(np.linalg.norm(z), 1e-12)
    u = xyz[roles["high_a"]] - xyz[a]
    v = xyz[roles["high_b"]] - xyz[b]
    u -= z * np.dot(z, u)
    v -= z * np.dot(z, v)
    u /= max(np.linalg.norm(u), 1e-12)
    v /= max(np.linalg.norm(v), 1e-12)
    signed_sin = float(np.dot(z, np.cross(u, v)))
    cos_phi = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return (z, u, v, abs(signed_sin), cos_phi)


def _matched_determinant_sign(result: dict, unit: tuple[int, ...]) -> float:
    for idx, predicted in enumerate(result.get("chiral axes", [])):
        if set(map(int, predicted)) == set(map(int, unit)):
            det = float(np.asarray(result["determinant"][idx]).reshape(-1)[0])
            return float(np.sign(det))
    return 0.0


def _unit_chi_and_shape(
    mol: Chem.Mol,
    unit: tuple[int, ...],
    result: dict,
    ranks: list[int],
    cip_ranks: list[int] | None = None,
    chiral_type: str | None = None,
) -> tuple[float, float, float]:
    """Compute chi plus parity-even |sin(phi)| and cos(phi).

    Two-endpoint units use the signed axial angle.  ChiDeK's ACMP also contains
    singleton "spiral atom and chain" annotations; those use the supplied
    determinant sign because they are not a two-endpoint stereogenic axis.
    """
    if len(unit) == 2:
        z, u, v, abs_sin, cos_phi = _axis_vectors(
            mol, (unit[0], unit[1]), ranks, cip_ranks=cip_ranks
        )
        signed_sin = float(np.dot(z, np.cross(u, v)))
        geometric_chi = float(np.sign(signed_sin))
        determinant_chi = _matched_determinant_sign(result, unit)
        if determinant_chi != 0.0:
            chi = determinant_chi
        else:
            chi = geometric_chi
            if chiral_type == "Chiral atom pair":
                chi = -chi
        return (chi, abs_sin, cos_phi)
    if len(unit) == 1:
        chi = _matched_determinant_sign(result, unit)
        return (chi, 1.0, 0.0)
    raise ValueError(f"Unsupported stereochemical unit: {unit}")


def _unit_relations(
    mol: Chem.Mol,
    unit: tuple[int, ...],
    graph_dist: np.ndarray,
    abs_sin: float,
    cos_phi: float,
    ranks: list[int] | None = None,
    cip_ranks: list[int] | None = None,
) -> np.ndarray:
    """CIP-directed, parity-even relation and path features for every atom."""
    xyz = _coords(mol).astype(np.float64)
    n = len(xyz)
    if ranks is None:
        ranks = list(
            Chem.CanonicalRankAtoms(
                mol, breakTies=False, includeChirality=False, includeIsotopes=True
            )
        )

    def path_statistics(path: tuple[int, ...]) -> tuple[float, float, float]:
        bonds = [
            mol.GetBondBetweenAtoms(int(left), int(right))
            for left, right in zip(path[:-1], path[1:])
        ]
        if not bonds:
            return (0.0, 0.0, 0.0)
        aromatic = np.mean([bond.GetIsAromatic() for bond in bonds])
        multiple = np.mean([bond.GetBondTypeAsDouble() > 1.0 for bond in bonds])
        ring = np.mean([bond.IsInRing() for bond in bonds])
        return (float(aromatic), float(multiple), float(ring))

    if len(unit) == 1:
        c = unit[0]
        dg = graph_dist[:, c]
        de = np.linalg.norm(xyz - xyz[c], axis=1)
        cip_ranks = _cip_ranks(mol) if cip_ranks is None else cip_ranks
        neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(c).GetNeighbors()]
        neighbors.sort(key=lambda idx: _priority_key(mol, idx, cip_ranks, ranks), reverse=True)
        relation = []
        for atom_index in range(n):
            path = (c,) if atom_index == c else rdmolops.GetShortestPath(mol, c, atom_index)
            aromatic, multiple, ring = path_statistics(path)
            branch = np.zeros(9, dtype=np.float64)
            if len(path) < 2:
                branch[8] = 1.0
            else:
                first = int(path[1])
                branch[neighbors.index(first) if first in neighbors[:4] else 3] = 1.0
            relation.append(
                [
                    dg[atom_index] / 10.0,
                    dg[atom_index] / 10.0,
                    de[atom_index] / 10.0,
                    de[atom_index] / 10.0,
                    de[atom_index] / 10.0,
                    0.0,
                    0.0,
                    0.0,
                    float(atom_index == c),
                    0.0,
                    abs_sin,
                    cos_phi,
                    0.0,
                    aromatic,
                    multiple,
                    ring,
                    *branch,
                    1.0,
                ]
            )
        result = np.asarray(relation, dtype=np.float32)
        assert result.shape == (n, REL_DIM)
        return result
    roles = _axis_roles(mol, (unit[0], unit[1]), ranks, cip_ranks=cip_ranks)
    a, b = (roles["a"], roles["b"])
    dg_a, dg_b = (graph_dist[:, a], graph_dist[:, b])
    de_a = np.linalg.norm(xyz - xyz[a], axis=1)
    de_b = np.linalg.norm(xyz - xyz[b], axis=1)
    z = xyz[b] - xyz[a]
    axis_length = max(np.linalg.norm(z), 1e-12)
    z /= axis_length
    mid = 0.5 * (xyz[a] + xyz[b])
    longitudinal = (xyz - mid) @ z
    radial = np.linalg.norm(xyz - mid - longitudinal[:, None] * z[None, :], axis=1)
    relation = []
    for atom_index in range(n):
        if dg_a[atom_index] < dg_b[atom_index]:
            endpoint, side_offset = (a, 0)
        elif dg_b[atom_index] < dg_a[atom_index]:
            endpoint, side_offset = (b, 4)
        elif longitudinal[atom_index] <= 0:
            endpoint, side_offset = (a, 0)
        else:
            endpoint, side_offset = (b, 4)
        path = (
            (endpoint,)
            if atom_index == endpoint
            else rdmolops.GetShortestPath(mol, endpoint, atom_index)
        )
        aromatic, multiple, ring = path_statistics(path)
        branch = np.zeros(9, dtype=np.float64)
        if atom_index in (a, b) or len(path) < 2:
            branch[8] = 1.0
        else:
            first = int(path[1])
            high = roles["high_a"] if endpoint == a else roles["high_b"]
            low = roles["low_a"] if endpoint == a else roles["low_b"]
            internal = roles["next_a"] if endpoint == a else roles["next_b"]
            role_index = (
                0 if first == high else 1 if first == low else 2 if first == internal else 3
            )
            branch[side_offset + role_index] = 1.0
        sigma = (
            -1.0
            if dg_a[atom_index] < dg_b[atom_index]
            else 1.0
            if dg_b[atom_index] < dg_a[atom_index]
            else 0.0
        )
        relation.append(
            [
                dg_a[atom_index] / 10.0,
                dg_b[atom_index] / 10.0,
                de_a[atom_index] / 10.0,
                de_b[atom_index] / 10.0,
                radial[atom_index] / 10.0,
                longitudinal[atom_index] / 10.0,
                abs(longitudinal[atom_index]) / 10.0,
                sigma,
                float(atom_index == a),
                float(atom_index == b),
                abs_sin,
                cos_phi,
                axis_length / 10.0,
                aromatic,
                multiple,
                ring,
                *branch,
                0.0,
            ]
        )
    result = np.asarray(relation, dtype=np.float32)
    assert result.shape == (n, REL_DIM)
    return result


def _phase_initialization(relations: np.ndarray) -> np.ndarray:
    """Chemically initialize an axial side field or central CIP-branch field."""
    if np.all(relations[:, -1] > 0.5):
        branch_angles = np.asarray(
            [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0, 0.0], dtype=np.float32
        )
        branch_phase = relations[:, 16:20] @ branch_angles
        graph_distance = 10.0 * relations[:, 0]
        return (branch_phase + 0.12 * graph_distance).astype(np.float32)
    angle = np.arctan2(np.abs(relations[:, 10]), relations[:, 11])
    side = 0.5 * relations[:, 7] * angle
    graph_distance = 10.0 * np.minimum(relations[:, 0], relations[:, 1])
    return (side + 0.12 * graph_distance).astype(np.float32)


def _unit_descriptor(
    mol: Chem.Mol,
    unit: tuple[int, ...],
    node_features: np.ndarray,
    abs_sin: float,
    cos_phi: float,
    ranks: list[int],
    cip_ranks: list[int] | None = None,
) -> np.ndarray:
    """Parity-even descriptor of ordered endpoints and CIP substituent roles."""
    xyz = _coords(mol).astype(np.float64)
    cip_ranks = _cip_ranks(mol) if cip_ranks is None else cip_ranks
    max_cip = max(max(cip_ranks), 1)
    if len(unit) == 2:
        roles = _axis_roles(mol, (unit[0], unit[1]), ranks, cip_ranks=cip_ranks)
        a, b = (roles["a"], roles["b"])
        role_atoms = [a, b, roles["high_a"], roles["low_a"], roles["high_b"], roles["low_b"]]
        z = xyz[b] - xyz[a]
        axis_length = max(np.linalg.norm(z), 1e-12)
        z /= axis_length
        substituents = role_atoms[2:]
        anchors = [a, a, b, b]
        singleton = 0.0
        candidate_counts = [len(roles["cand_a"]), len(roles["cand_b"])]
    else:
        c = unit[0]
        neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(c).GetNeighbors()]
        neighbors.sort(key=lambda idx: _priority_key(mol, idx, cip_ranks, ranks), reverse=True)
        substituents = (neighbors + [None] * 4)[:4]
        role_atoms = [c, None, *substituents]
        anchors = [c] * 4
        z = np.zeros(3, dtype=np.float64)
        axis_length = 0.0
        singleton = 1.0
        candidate_counts = [len(neighbors), 0]
    atom_blocks = [
        node_features[index] if index is not None else np.zeros(NODE_FEATURE_DIM)
        for index in role_atoms
    ]
    scalar = [singleton, axis_length / 10.0, abs_sin, cos_phi]
    vectors = []
    for substituent, anchor in zip(substituents, anchors):
        if substituent is None:
            scalar.extend([0.0] * 5)
            vectors.append(None)
            continue
        vector = xyz[substituent] - xyz[anchor]
        length = np.linalg.norm(vector)
        longitudinal = float(np.dot(vector, z)) if axis_length else 0.0
        radial = np.linalg.norm(vector - longitudinal * z) if axis_length else length
        bond = mol.GetBondBetweenAtoms(int(anchor), int(substituent))
        scalar.extend(
            [
                1.0,
                bond.GetBondTypeAsDouble() / 3.0,
                length / 3.0,
                radial / 3.0,
                cip_ranks[substituent] / max_cip,
            ]
        )
        vectors.append(vector / max(length, 1e-12))

    def vector_dot(left: int, right: int) -> float:
        if vectors[left] is None or vectors[right] is None:
            return 0.0
        return float(np.dot(vectors[left], vectors[right]))

    scalar.extend([vector_dot(0, 1), vector_dot(2, 3)])
    scalar.extend(
        [
            mol.GetAtomWithIdx(role_atoms[0]).GetDegree() / 6.0,
            mol.GetAtomWithIdx(role_atoms[1]).GetDegree() / 6.0
            if role_atoms[1] is not None
            else 0.0,
            candidate_counts[0] / 4.0,
            candidate_counts[1] / 4.0,
        ]
    )
    descriptor = np.concatenate([*atom_blocks, np.asarray(scalar)]).astype(np.float32)
    assert descriptor.shape == (UNIT_DESC_DIM,), descriptor.shape
    return descriptor


def build_sample(
    row: dict,
    result: dict,
    label_row: pd.Series,
    optical_by_id: dict[int, float],
    units_override: list[tuple[int, ...]] | None = None,
) -> dict:
    mol = row["rdkit_mol"]
    xyz = _coords(mol)
    n = mol.GetNumAtoms()
    base_id, is_mirror = _id_parts(row["id"])
    graph_dist = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float32)
    euclidean = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adjacency = np.asarray(rdmolops.GetAdjacencyMatrix(mol), dtype=np.float32)
    pair = np.stack(
        [np.clip(graph_dist, 0, 15) / 15.0, np.clip(euclidean, 0, 20) / 20.0, adjacency], axis=-1
    ).astype(np.float32)
    atoms = list(mol.GetAtoms())
    node = atom_features(atoms, mol, include_cip=False)
    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False, includeIsotopes=True)
    )
    cip_ranks = _cip_ranks(mol)
    curated_units = [tuple(map(int, x)) for x in ast.literal_eval(label_row["label"])]
    if units_override is None:
        units = curated_units
    else:
        units = sorted(
            [tuple(sorted(map(int, unit))) for unit in units_override],
            key=lambda unit: (len(unit), unit),
        )
    unit_rel, unit_desc, phase_init, chi, rho = ([], [], [], [], [])
    local_chi = np.zeros((n, 1), dtype=np.float32)
    for unit in units:
        one_chi, abs_sin, cos_phi = _unit_chi_and_shape(
            mol, unit, result, ranks, cip_ranks=cip_ranks, chiral_type=str(label_row["chiral_type"])
        )
        if one_chi == 0.0:
            raise ValueError(f"Could not determine chi for molecule {row['id']} unit {unit}")
        relations = _unit_relations(
            mol, unit, graph_dist, abs_sin, cos_phi, ranks=ranks, cip_ranks=cip_ranks
        )
        unit_rel.append(relations)
        unit_desc.append(
            _unit_descriptor(mol, unit, node, abs_sin, cos_phi, ranks, cip_ranks=cip_ranks)
        )
        phase_init.append(_phase_initialization(relations))
        chi.append(one_chi)
        rho.append(1.0)
        for atom_idx in unit:
            local_chi[atom_idx, 0] += one_chi
    base_label = int(optical_by_id[base_id] > 0.0)
    label = 1 - base_label if is_mirror else base_label
    if unit_rel:
        unit_rel_array = np.stack(unit_rel).astype(np.float32)
        unit_desc_array = np.stack(unit_desc).astype(np.float32)
        phase_init_array = np.stack(phase_init).astype(np.float32)
    else:
        unit_rel_array = np.zeros((0, n, REL_DIM), dtype=np.float32)
        unit_desc_array = np.zeros((0, UNIT_DESC_DIM), dtype=np.float32)
        phase_init_array = np.zeros((0, n), dtype=np.float32)
    return {
        "id": row["id"],
        "base_id": base_id,
        "node": node,
        "pair": pair,
        "unit_rel": unit_rel_array,
        "unit_desc": unit_desc_array,
        "phase_init": phase_init_array,
        "chi": np.asarray(chi, dtype=np.float32),
        "rho": np.asarray(rho, dtype=np.float32),
        "local_chi": local_chi,
        "label": label,
        "annotation_source": "curated" if units_override is None else "automatic_chiralfinder",
        "curated_units": sorted(
            [tuple(sorted(unit)) for unit in curated_units], key=lambda unit: (len(unit), unit)
        ),
        "input_units": units,
    }


def load_acmp(data_dir: Path, annotation_source: str = "curated") -> tuple[list[dict], dict]:
    if annotation_source not in {"curated", "automatic"}:
        raise ValueError(f"Unknown ACMP annotation source: {annotation_source}")
    with open(data_dir / "hct_ecd_axial.pkl", "rb") as handle:
        raw = pickle.load(handle)
    with open(data_dir / "hct_ecd_axial_res.pkl", "rb") as handle:
        results = pickle.load(handle)
    with open(data_dir / "ecd_axial_index_split.pkl", "rb") as handle:
        split = pickle.load(handle)
    labels = _read_xlsx_first_sheet(data_dir / "axial_650.xlsx")
    optical = pd.read_csv(data_dir / "optical_rotation_589nm.csv")
    optical_by_id = dict(zip(optical["id"].astype(int), optical["OR_589nm"].astype(float)))
    samples = []
    for row, result in zip(raw, results):
        base_id, _ = _id_parts(row["id"])
        units_override = None
        if annotation_source == "automatic":
            units_override = [tuple(map(int, unit)) for unit in result.get("chiral axes", [])]
        samples.append(
            build_sample(
                row, result, labels.iloc[base_id], optical_by_id, units_override=units_override
            )
        )
    return (samples, split)


def verify_dataset(samples: list[dict], split: dict) -> dict:
    by_base: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        by_base[sample["base_id"]].append(sample)
    complete = [pair for pair in by_base.values() if len(pair) == 2]
    chi_inverse = all((np.array_equal(pair[0]["chi"], -pair[1]["chi"]) for pair in complete))
    even_equal = all(
        (
            np.allclose(pair[0]["node"], pair[1]["node"], atol=1e-06)
            and np.allclose(pair[0]["pair"], pair[1]["pair"], atol=1e-06)
            and np.allclose(pair[0]["unit_rel"], pair[1]["unit_rel"], atol=1e-06)
            and np.allclose(pair[0]["unit_desc"], pair[1]["unit_desc"], atol=1e-06)
            and np.allclose(pair[0]["phase_init"], pair[1]["phase_init"], atol=1e-06)
            for pair in complete
        )
    )
    split_bases = {
        key: {samples[int(i)]["base_id"] for i in indices} for key, indices in split.items()
    }
    overlap = {
        f"{a}:{b}": len(split_bases[a] & split_bases[b])
        for pos, a in enumerate(split_bases)
        for b in list(split_bases)[pos + 1 :]
    }
    annotation_matches: dict[str, int] = defaultdict(int)
    for sample in samples:
        curated = set(sample.get("curated_units", []))
        observed = set(sample.get("input_units", []))
        if observed == curated:
            category = "exact"
        elif observed > curated:
            category = "automatic_superset"
        elif observed < curated:
            category = "automatic_subset"
        else:
            category = "different"
        annotation_matches[category] += 1
    return {
        "molecules": len(samples),
        "enantiomer_pairs": len(complete),
        "annotation_source": samples[0].get("annotation_source", "unknown"),
        "molecules_without_detected_units": sum((len(sample["chi"]) == 0 for sample in samples)),
        "mean_input_units": float(np.mean([len(sample["chi"]) for sample in samples])),
        "automatic_exact_match_molecules": sum(
            (sample.get("input_units") == sample.get("curated_units") for sample in samples)
        ),
        "annotation_match_counts": dict(annotation_matches),
        "all_chi_vectors_invert": chi_inverse,
        "all_parity_even_inputs_match": even_equal,
        "split_pair_overlap": overlap,
    }
