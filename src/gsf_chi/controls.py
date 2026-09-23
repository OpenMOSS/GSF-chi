from __future__ import annotations

from collections import defaultdict

import numpy as np


def _stable_support_rng(
    base_id: int, unit_index: int, support_seed: int, salt: int
) -> np.random.Generator:
    """Return a reproducible RNG shared by both members of an enantiomer pair."""
    return np.random.default_rng(
        np.random.SeedSequence([int(support_seed), int(base_id), int(unit_index), int(salt)])
    )


def attach_gsf_support_masks(
    samples: list[dict], mode: str, fraction: float = 0.0, support_seed: int = 20260828
) -> dict:
    """Attach fixed random or distance-matched all-pair training supports.

    A zero ``fraction`` requests exactly the number of directed entries used by
    the query-anchor scope for each molecule and stereogenic unit.  Positive
    fractions request that fraction of all valid directed atom pairs.  Masks
    depend only on parity-even graph quantities and ``base_id`` so an
    enantiomer pair receives exactly the same support.
    """
    if mode not in {"random", "distance"}:
        raise ValueError(f"Fixed support masks do not implement mode {mode!r}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("GSF support fraction must be in [0, 1]")
    retained_fractions: list[float] = []
    query_overlaps: list[float] = []
    exact_distance_matches: list[float] = []
    selected_counts: list[int] = []
    for sample in samples:
        n_atoms = len(sample["node"])
        n_units = len(sample["chi"])
        mask = np.zeros((n_units, n_atoms, n_atoms), dtype=np.float32)
        graph_distance = np.rint(sample["pair"][:, :, 0] * 15.0).astype(np.int64)
        for unit_index in range(n_units):
            anchor = sample["unit_rel"][unit_index, :, -2] > 0.5
            query_support = np.broadcast_to(anchor[:, None], (n_atoms, n_atoms)).reshape(-1)
            if fraction > 0.0:
                target_count = int(round(fraction * n_atoms * n_atoms))
                target_count = min(max(target_count, 1), n_atoms * n_atoms)
            else:
                target_count = int(query_support.sum())
            chosen = np.zeros(n_atoms * n_atoms, dtype=bool)
            rng = _stable_support_rng(
                int(sample["base_id"]), unit_index, support_seed, 11 if mode == "random" else 29
            )
            if mode == "random" or fraction > 0.0:
                selected = rng.choice(n_atoms * n_atoms, size=target_count, replace=False)
                chosen[selected] = True
            else:
                distance_flat = graph_distance.reshape(-1)
                for distance in np.unique(distance_flat[query_support]):
                    needed = int(np.sum(query_support & (distance_flat == distance)))
                    candidates = np.flatnonzero(distance_flat == distance)
                    selected = rng.choice(candidates, size=needed, replace=False)
                    chosen[selected] = True
                target_hist = np.bincount(
                    distance_flat[query_support], minlength=int(distance_flat.max()) + 1
                )
                chosen_hist = np.bincount(
                    distance_flat[chosen], minlength=int(distance_flat.max()) + 1
                )
                exact_distance_matches.append(float(np.array_equal(target_hist, chosen_hist)))
            if int(chosen.sum()) != target_count:
                raise AssertionError("Fixed GSF support has the wrong entry count")
            mask[unit_index] = chosen.reshape(n_atoms, n_atoms)
            retained_fractions.append(target_count / float(n_atoms * n_atoms))
            query_overlaps.append(float(np.sum(chosen & query_support)) / max(target_count, 1))
            selected_counts.append(target_count)
        sample["gsf_support_mask"] = mask
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample["base_id"])].append(sample["gsf_support_mask"])
    complete_pairs = [masks for masks in grouped.values() if len(masks) == 2]
    pair_masks_equal = all((np.array_equal(masks[0], masks[1]) for masks in complete_pairs))
    if not pair_masks_equal:
        raise AssertionError("An enantiomer pair received different support masks")
    return {
        "mode": mode,
        "requested_fraction": float(fraction),
        "support_seed": int(support_seed),
        "mean_retained_fraction": float(np.mean(retained_fractions)),
        "mean_selected_entries_per_unit": float(np.mean(selected_counts)),
        "mean_overlap_with_query_anchor": float(np.mean(query_overlaps)),
        "exact_distance_histogram_match_fraction": float(np.mean(exact_distance_matches))
        if exact_distance_matches
        else None,
        "complete_enantiomer_pairs": len(complete_pairs),
        "all_pair_masks_equal": pair_masks_equal,
    }


def _stable_protocol_order(base_ids: list[int], seed: int, salt: int) -> list[int]:
    """Order pair identities without consulting any molecular target."""

    def key(base_id: int) -> int:
        state = np.random.SeedSequence([int(seed), int(salt), int(base_id)]).generate_state(
            1, dtype=np.uint64
        )
        return int(state[0])

    return sorted(base_ids, key=lambda base_id: (key(base_id), base_id))


def _group_indices_by_base(samples: list[dict], indices: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for index in indices:
        grouped.setdefault(int(samples[int(index)]["base_id"]), []).append(int(index))
    for base_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(
                f"Low-data protocols require complete pairs; base {base_id} has {len(members)} members"
            )
        members.sort(key=lambda index: str(samples[index]["id"]))
    return grouped


def apply_training_protocol(
    samples: list[dict], split: dict, protocol: str, pair_fraction: float, protocol_seed: int
) -> tuple[dict, dict]:
    """Create target-blind low-data and one-enantiomer supervision contracts."""
    if protocol not in {"full", "pair_fraction", "one_side"}:
        raise ValueError(f"Unknown training protocol: {protocol}")
    if not 0.0 < pair_fraction <= 1.0:
        raise ValueError("Training pair fraction must be in (0, 1]")
    transformed = {key: [int(index) for index in indices] for key, indices in split.items()}
    audit: dict[str, object] = {
        "protocol": protocol,
        "pair_fraction": float(pair_fraction),
        "protocol_seed": int(protocol_seed),
        "selection_uses_mirror_labels": protocol != "one_side",
    }
    train_grouped = _group_indices_by_base(samples, transformed["train_index"])
    val_grouped = _group_indices_by_base(samples, transformed["val_index"])
    if protocol == "pair_fraction":
        ordered = _stable_protocol_order(list(train_grouped), protocol_seed, salt=101)
        keep_pairs = max(1, int(round(pair_fraction * len(ordered))))
        selected_bases = set(ordered[:keep_pairs])
        transformed["train_index"] = [
            index for base_id in ordered[:keep_pairs] for index in train_grouped[base_id]
        ]
        audit.update(
            train_pair_identities=len(selected_bases),
            train_molecules=len(transformed["train_index"]),
        )
    elif protocol == "one_side":
        observed_train: list[int] = []
        withheld_train: list[int] = []
        observed_val: list[int] = []
        withheld_val: list[int] = []
        for grouped, observed, withheld, salt in (
            (train_grouped, observed_train, withheld_train, 211),
            (val_grouped, observed_val, withheld_val, 307),
        ):
            for base_id in sorted(grouped):
                members = grouped[base_id]
                selector = _stable_protocol_order([0, 1], protocol_seed + base_id, salt=salt)[0]
                observed.append(members[selector])
                withheld.append(members[1 - selector])
        transformed["train_index"] = observed_train
        transformed["val_index"] = observed_val
        transformed["mirror_completion_index"] = withheld_train
        transformed["mirror_completion_pair_index"] = sorted(observed_train + withheld_train)
        transformed["validation_mirror_index"] = withheld_val
        audit.update(
            train_pair_identities=len(train_grouped),
            train_molecules=len(observed_train),
            withheld_train_mirrors=len(withheld_train),
            validation_molecules=len(observed_val),
            withheld_validation_mirrors=len(withheld_val),
            mirror_member_selection="target-blind deterministic hash of pair identity",
        )
    else:
        audit.update(
            train_pair_identities=len(train_grouped),
            train_molecules=len(transformed["train_index"]),
        )
    audit["test_molecules"] = len(transformed["test_index"])
    audit["test_pair_identities"] = len(_group_indices_by_base(samples, transformed["test_index"]))
    return (transformed, audit)
