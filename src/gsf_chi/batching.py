from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from gsf_chi.features import REL_DIM, UNIT_DESC_DIM


class SampleDataset(Dataset):
    def __init__(self, samples: list[dict], indices: Iterable[int]):
        self.samples = [samples[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def collate(samples: list[dict]) -> dict:
    bsz = len(samples)
    nmax = max((len(x["node"]) for x in samples))
    smax = max((len(x["chi"]) for x in samples))
    fdim = samples[0]["node"].shape[-1]
    node = np.zeros((bsz, nmax, fdim), np.float32)
    pair = np.zeros((bsz, nmax, nmax, 3), np.float32)
    rel = np.zeros((bsz, smax, nmax, REL_DIM), np.float32)
    unit_desc = np.zeros((bsz, smax, UNIT_DESC_DIM), np.float32)
    phase_init = np.zeros((bsz, smax, nmax), np.float32)
    chi = np.zeros((bsz, smax), np.float32)
    rho = np.zeros((bsz, smax), np.float32)
    local_chi = np.zeros((bsz, nmax, 1), np.float32)
    node_mask = np.zeros((bsz, nmax), bool)
    unit_mask = np.zeros((bsz, smax), bool)
    for b, sample in enumerate(samples):
        n, s = (len(sample["node"]), len(sample["chi"]))
        node[b, :n] = sample["node"]
        pair[b, :n, :n] = sample["pair"]
        rel[b, :s, :n] = sample["unit_rel"]
        unit_desc[b, :s] = sample["unit_desc"]
        phase_init[b, :s, :n] = sample["phase_init"]
        chi[b, :s] = sample["chi"]
        rho[b, :s] = sample["rho"]
        local_chi[b, :n] = sample["local_chi"]
        node_mask[b, :n] = True
        unit_mask[b, :s] = True
    batch = {
        "node": torch.from_numpy(node),
        "pair": torch.from_numpy(pair),
        "unit_rel": torch.from_numpy(rel),
        "unit_desc": torch.from_numpy(unit_desc),
        "phase_init": torch.from_numpy(phase_init),
        "chi": torch.from_numpy(chi),
        "rho": torch.from_numpy(rho),
        "local_chi": torch.from_numpy(local_chi),
        "node_mask": torch.from_numpy(node_mask),
        "unit_mask": torch.from_numpy(unit_mask),
        "label": torch.tensor([x["label"] for x in samples], dtype=torch.long),
        "base_id": torch.tensor([x["base_id"] for x in samples], dtype=torch.long),
        "id": [x["id"] for x in samples],
    }
    has_support_mask = ["gsf_support_mask" in sample for sample in samples]
    if any(has_support_mask) and (not all(has_support_mask)):
        raise ValueError("Cannot mix samples with and without GSF support masks")
    if all(has_support_mask):
        support_mask = np.zeros((bsz, smax, nmax, nmax), np.float32)
        for b, sample in enumerate(samples):
            n, s = (len(sample["node"]), len(sample["chi"]))
            one_mask = np.asarray(sample["gsf_support_mask"], dtype=np.float32)
            if one_mask.shape != (s, n, n):
                raise ValueError(
                    f"GSF support mask must have shape [units,atoms,atoms], got {one_mask.shape} for {sample['id']}"
                )
            support_mask[b, :s, :n, :n] = one_mask
        batch["gsf_support_mask"] = torch.from_numpy(support_mask)
    return batch
