"""Regression tests for sparse-support and low-data reviewer controls."""

from __future__ import annotations

import copy

import numpy as np
import torch

import gsf_chi.attention as gsf_attention
import gsf_chi.controls as gsf_controls
import gsf_chi.data.ecd as gsf_data_ecd
import gsf_chi.features as gsf_features


def _sample(base_id: int, mirror: bool, n_atoms: int = 5) -> dict:
    pair = np.zeros((n_atoms, n_atoms, 3), dtype=np.float32)
    distance = np.abs(np.arange(n_atoms)[:, None] - np.arange(n_atoms)[None, :])
    pair[..., 0] = distance / 15.0
    unit_rel = np.zeros((1, n_atoms, gsf_features.REL_DIM), dtype=np.float32)
    unit_rel[0, 0, -2] = 1.0
    return {
        "id": f"mol_{('-' if mirror else '')}{base_id}",
        "base_id": base_id,
        "node": np.zeros((n_atoms, 52), dtype=np.float32),
        "pair": pair,
        "unit_rel": unit_rel,
        "unit_desc": np.zeros((1, gsf_features.UNIT_DESC_DIM), dtype=np.float32),
        "phase_init": np.zeros((1, n_atoms), dtype=np.float32),
        "chi": np.asarray([-1.0 if mirror else 1.0], dtype=np.float32),
        "rho": np.ones(1, dtype=np.float32),
        "local_chi": np.zeros((n_atoms, 1), dtype=np.float32),
        "label": int(mirror),
        "peak_num": 1,
        "peak_position": np.asarray([3], dtype=np.int64),
        "peak_height": np.asarray([int(mirror)], dtype=np.int64),
    }


def test_fixed_support_is_pair_shared_and_count_matched() -> None:
    for mode in ("random", "distance"):
        samples = [_sample(7, False), _sample(7, True)]
        audit = gsf_controls.attach_gsf_support_masks(
            samples, mode=mode, fraction=0.0, support_seed=17
        )
        assert audit["all_pair_masks_equal"]
        assert np.array_equal(samples[0]["gsf_support_mask"], samples[1]["gsf_support_mask"])
        assert int(samples[0]["gsf_support_mask"].sum()) == 5
        if mode == "distance":
            assert audit["exact_distance_histogram_match_fraction"] == 1.0


def test_collate_preserves_fixed_support() -> None:
    samples = [_sample(7, False), _sample(7, True)]
    gsf_controls.attach_gsf_support_masks(samples, "random", 0.0, 19)
    batch = gsf_data_ecd.collate_ecd(samples)
    assert batch["gsf_support_mask"].shape == (2, 1, 5, 5)
    assert torch.equal(batch["gsf_support_mask"][0], batch["gsf_support_mask"][1])


def test_learned_topk_exactly_matches_anchor_budget() -> None:
    layer = gsf_attention.GSFChiRoPELayer(
        d_model=24,
        n_heads=4,
        rel_dim=gsf_features.REL_DIM,
        unit_desc_dim=gsf_features.UNIT_DESC_DIM,
        n_chiral_heads=2,
        dropout=0.0,
        chiral_scale_init=5.0,
        gsf_support_mode="learned_topk",
        gsf_support_fraction=0.0,
    )
    score = torch.rand(2, 1, 5, 5)
    anchor = torch.zeros(2, 1, 5)
    anchor[:, :, 0] = 1.0
    node_mask = torch.ones(2, 5, dtype=torch.bool)
    unit_mask = torch.ones(2, 1, dtype=torch.bool)
    support = layer._learned_topk_support(score, anchor, node_mask, unit_mask)
    assert torch.equal(support.sum(dim=(-1, -2)), torch.full((2, 1), 5.0))


def test_one_side_protocol_never_uses_withheld_mirror_for_selection() -> None:
    samples = []
    for base_id in range(8):
        samples.extend([_sample(base_id, False), _sample(base_id, True)])
    split = {
        "train_index": list(range(0, 8)),
        "val_index": list(range(8, 12)),
        "test_index": list(range(12, 16)),
    }
    transformed, audit = gsf_controls.apply_training_protocol(
        copy.deepcopy(samples), split, "one_side", 1.0, 23
    )
    assert not audit["selection_uses_mirror_labels"]
    assert len(transformed["train_index"]) == 4
    assert len(transformed["mirror_completion_index"]) == 4
    assert len(transformed["val_index"]) == 2
    assert len(transformed["validation_mirror_index"]) == 2
    assert set(transformed["train_index"]).isdisjoint(transformed["mirror_completion_index"])
    assert set(transformed["val_index"]).isdisjoint(transformed["validation_mirror_index"])


def test_pair_fraction_keeps_complete_pairs() -> None:
    samples = []
    for base_id in range(8):
        samples.extend([_sample(base_id, False), _sample(base_id, True)])
    split = {
        "train_index": list(range(0, 8)),
        "val_index": list(range(8, 12)),
        "test_index": list(range(12, 16)),
    }
    transformed, _ = gsf_controls.apply_training_protocol(samples, split, "pair_fraction", 0.5, 29)
    kept_base_ids = [samples[index]["base_id"] for index in transformed["train_index"]]
    assert len(kept_base_ids) == 4
    assert all((kept_base_ids.count(base_id) == 2 for base_id in set(kept_base_ids)))
