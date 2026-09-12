from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

import gsf_chi.batching as batching
import gsf_chi.data.axial as data_axial
from gsf_chi.ecd import N_PEAK_SLOTS, N_POSITION_CLASSES


def load_ecd_samples(data_dir: Path, annotation_source: str = "curated") -> tuple[list[dict], dict]:
    samples, split = data_axial.load_acmp(data_dir, annotation_source=annotation_source)
    with open(data_dir / "hct_ecd_axial.pkl", "rb") as handle:
        raw = pickle.load(handle)
    enriched = []
    for sample, row in zip(samples, raw):
        item = dict(sample)
        item["peak_num"] = int(row["peak_num"])
        item["peak_position"] = np.asarray(row["peak_position"][:N_PEAK_SLOTS], dtype=np.int64)
        item["peak_height"] = np.asarray(row["peak_height"][:N_PEAK_SLOTS], dtype=np.int64)
        enriched.append(item)
    return (enriched, split)


def collate_ecd(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty ECD batch")
    n_slots = len(samples[0]["peak_position"])
    if any((len(s["peak_position"]) != n_slots for s in samples)):
        raise ValueError("An ECD batch must use one peak-slot count")
    batch = batching.collate(samples)
    positions = np.full((len(samples), n_slots), -1, dtype=np.int64)
    heights = np.full((len(samples), n_slots), -1, dtype=np.int64)
    for idx, sample in enumerate(samples):
        npos = min(len(sample["peak_position"]), n_slots)
        nheight = min(len(sample["peak_height"]), n_slots)
        positions[idx, :npos] = sample["peak_position"][:npos]
        heights[idx, :nheight] = sample["peak_height"][:nheight]
    batch["peak_num"] = torch.tensor([x["peak_num"] for x in samples], dtype=torch.long)
    batch["peak_position"] = torch.from_numpy(positions)
    batch["peak_height"] = torch.from_numpy(heights)
    if samples and all(("spectrum_abs" in sample for sample in samples)):
        spectrum_abs = np.stack(
            [np.asarray(sample["spectrum_abs"], dtype=np.float32) for sample in samples]
        )
        if spectrum_abs.shape != (len(samples), N_POSITION_CLASSES):
            raise ValueError(
                f"spectrum_abs must have shape [batch,{N_POSITION_CLASSES}], got {spectrum_abs.shape}"
            )
        peak_mask = np.zeros_like(spectrum_abs, dtype=np.float32)
        for sample_index, sample in enumerate(samples):
            peak_count = min(int(sample["peak_num"]), n_slots)
            peak_indices = np.asarray(sample["peak_position"][:peak_count], dtype=np.int64)
            if len(peak_indices):
                peak_mask[sample_index, peak_indices] = 1.0
        batch["spectrum_abs"] = torch.from_numpy(spectrum_abs)
        batch["spectrum_peak_mask"] = torch.from_numpy(peak_mask)
    return batch


def dataset_audit(samples: list[dict]) -> dict:
    by_base: dict[int, list[dict]] = {}
    for sample in samples:
        by_base.setdefault(sample["base_id"], []).append(sample)
    pairs = [values for values in by_base.values() if len(values) == 2]
    annotation_matches: dict[str, int] = {}
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
        annotation_matches[category] = annotation_matches.get(category, 0) + 1
    return {
        "molecules": len(samples),
        "pairs": len(pairs),
        "annotation_source": samples[0].get("annotation_source", "unknown"),
        "molecules_without_detected_units": sum((len(sample["chi"]) == 0 for sample in samples)),
        "mean_input_units": float(np.mean([len(sample["chi"]) for sample in samples])),
        "automatic_exact_match_molecules": sum(
            (sample.get("input_units") == sample.get("curated_units") for sample in samples)
        ),
        "annotation_match_counts": annotation_matches,
        "number_equal_in_all_pairs": all((a["peak_num"] == b["peak_num"] for a, b in pairs)),
        "position_equal_in_all_pairs": all(
            (np.array_equal(a["peak_position"], b["peak_position"]) for a, b in pairs)
        ),
        "height_complement_in_all_nonempty_pairs": all(
            (
                np.all(a["peak_height"][: a["peak_num"]] + b["peak_height"][: b["peak_num"]] == 1)
                for a, b in pairs
                if a["peak_num"] > 0
            )
        ),
    }
