"""Shared split definitions for the public central-chirality ECD subset.

The pair-safe protocol is stored in ``gsf_central_ecd_cache.pkl``.  The
ECDFormer protocol deliberately shuffles individual augmented molecules, so
members of an enantiomer pair can occur in different partitions.  Keeping the
implementation here prevents ChiDeK and GSF-chi from silently using slightly
different recreations of the latter split.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PAIR_SAFE_SPLIT = "pair-safe-80-10-10"
ECDFORMER_9055_SPLIT = "ecdformer-90-5-5"
PHYSECD_RANDOM_9055_SPLIT = "physecd-random-90-5-5"
SPLIT_PROTOCOLS = (PAIR_SAFE_SPLIT, ECDFORMER_9055_SPLIT, PHYSECD_RANDOM_9055_SPLIT)


def ecdformer_9055_indices(
    num_samples: int, seed: int = 1101
) -> tuple[dict[str, list[int]], list[int]]:
    """Exactly reproduce ECDFormer's released sample-level ``fixed`` split."""
    order = np.arange(num_samples, dtype=np.int64)
    random_state = np.random.RandomState(seed)
    random_state.shuffle(order)
    train_num = int(num_samples * 0.9)
    val_num = int(num_samples * 0.05)
    test_num = int(num_samples * 0.05)
    split = {
        "train_index": order[:train_num].tolist(),
        "val_index": order[train_num : train_num + val_num].tolist(),
        "test_index": order[num_samples - test_num :].tolist(),
    }
    used = set().union(*(set(indices) for indices in split.values()))
    unused = [index for index in order.tolist() if index not in used]
    return (split, unused)


def physecd_random_9055_indices(
    samples: Sequence[dict], seed: int = 42
) -> tuple[dict[str, list[int]], list[int]]:
    """Create a random parent-level 90/5/5 split with intact enantiomer pairs.

    This follows the public *structure* of PhysECD's split: parent identities
    are partitioned first and both configurations remain inside that split.
    PhysECD v1 does not release its exact membership list, so this protocol is
    deliberately named ``random`` and must not be presented as their exact
    split.  The final split receives the rounding remainder; for 10,004
    parents this gives the paper's 9,003/500/501 counts.
    """
    grouped: dict[int, list[int]] = {}
    for sample_index, sample in enumerate(samples):
        grouped.setdefault(int(sample["base_id"]), []).append(sample_index)
    malformed = {base_id: members for base_id, members in grouped.items() if len(members) != 2}
    if malformed:
        preview = list(sorted(malformed.items()))[:5]
        raise ValueError(
            f"PhysECD-style splitting requires exactly two configurations per parent; malformed examples: {preview}"
        )
    parent_ids = np.asarray(sorted(grouped), dtype=np.int64)
    random_state = np.random.RandomState(seed)
    random_state.shuffle(parent_ids)
    train_num = int(len(parent_ids) * 0.9)
    val_num = int(len(parent_ids) * 0.05)
    parent_split = {
        "train_index": parent_ids[:train_num],
        "val_index": parent_ids[train_num : train_num + val_num],
        "test_index": parent_ids[train_num + val_num :],
    }
    split = {
        name: [
            sample_index
            for base_id in ids.tolist()
            for sample_index in sorted(grouped[int(base_id)])
        ]
        for name, ids in parent_split.items()
    }
    return (split, [])


def apply_split_protocol(
    samples: Sequence[dict],
    pair_safe_split: dict[str, list[int]],
    protocol: str,
    split_seed: int = 1101,
) -> tuple[dict[str, list[int]], dict]:
    """Return a split plus an auditable protocol contract."""
    if protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"Unknown split protocol {protocol!r}; expected {SPLIT_PROTOCOLS}")
    if protocol == ECDFORMER_9055_SPLIT:
        split, unused = ecdformer_9055_indices(len(samples), split_seed)
        unit = "augmented_molecule_sample"
        seed = int(split_seed)
        ratio = [0.9, 0.05, 0.05]
        test_mode = "fixed"
    elif protocol == PHYSECD_RANDOM_9055_SPLIT:
        split, unused = physecd_random_9055_indices(samples, split_seed)
        unit = "parent_molecule"
        seed = int(split_seed)
        ratio = [0.9, 0.05, 0.05]
        test_mode = "reconstructed_random"
    else:
        split = {name: list(map(int, indices)) for name, indices in pair_safe_split.items()}
        unused = []
        unit = "enantiomer_pair"
        seed = 42
        ratio = [0.8, 0.1, 0.1]
        test_mode = None
    pair_ids = {
        name: {int(samples[index]["base_id"]) for index in indices}
        for name, indices in split.items()
    }
    overlap = {
        "train_val": len(pair_ids["train_index"] & pair_ids["val_index"]),
        "train_test": len(pair_ids["train_index"] & pair_ids["test_index"]),
        "val_test": len(pair_ids["val_index"] & pair_ids["test_index"]),
    }
    contract = {
        "protocol": protocol,
        "unit": unit,
        "seed": seed,
        "ratio": ratio,
        "test_mode": test_mode,
        "source_samples": len(samples),
        "split_samples": {
            name.removesuffix("_index"): len(indices) for name, indices in split.items()
        },
        "unused_samples": len(unused),
        "unused_indices": unused,
        "enantiomer_pair_overlap": overlap,
        "pair_safe": not any(overlap.values()),
        "height_label_semantics": "chemically_complementary_enantiomers",
    }
    return (split, contract)
