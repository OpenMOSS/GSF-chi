"""Small RDKit atom featurizer used by the standalone GSF-chi models."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

ATOM_TYPES = ["H", "C", "B", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
FORMAL_CHARGES = [-1, -2, 1, 2, 0]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
NUM_HS = [0, 1, 2, 3, 4]
LOCAL_CHIRAL_TAGS = [0, 1, 2, 3]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]


def _one_hot(value, options):
    encoding = [0] * (len(options) + 1)
    encoding[options.index(value) if value in options else -1] = 1
    return encoding


def atom_features(atoms, molecule, include_cip: bool = True) -> np.ndarray:
    """Return the 52-dimensional atom features used in the experiments.

    Setting include_cip=False removes atom-local R/S labels. GSF-chi uses
    this setting so handedness enters only through the signed field.
    """
    cip = (
        dict(
            Chem.FindMolChiralCenters(
                molecule, force=True, includeUnassigned=True, useLegacyImplementation=False
            )
        )
        if include_cip
        else {}
    )
    rows = []
    for atom in atoms:
        row = _one_hot(atom.GetSymbol(), ATOM_TYPES)
        row += _one_hot(atom.GetTotalDegree(), DEGREES)
        row += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        row += _one_hot(atom.GetTotalNumHs(), NUM_HS)
        row += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        row += [int(atom.GetIsAromatic()), atom.GetMass() * 0.01]
        cip_value = cip.get(atom.GetIdx())
        cip_class = 1 if cip_value == "R" else 2 if cip_value == "S" else 0
        row += _one_hot(cip_class, [0, 1, 2])
        row += _one_hot(atom.GetChiralTag(), LOCAL_CHIRAL_TAGS)
        rows.append(row)
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 52:
        raise ValueError(f"unexpected atom feature shape: {result.shape}")
    return result


NODE_FEATURE_DIM = 52
REL_DIM = 26
UNIT_DESC_DIM = 6 * NODE_FEATURE_DIM + 30
