"""Keep repeated dataset property tests independent without rebuilding each graph."""

import copy
from functools import lru_cache

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session", autouse=True)
def cached_data_loaders():
    from gsf_chi.data import axial, ecd

    originals = []
    for module, name in [(axial, "load_acmp"), (ecd, "load_ecd_samples")]:
        original = getattr(module, name)
        cached = lru_cache(maxsize=4)(original)

        def fresh(*args, _cached=cached, **kwargs):
            return copy.deepcopy(_cached(*args, **kwargs))

        originals.append((module, name, original))
        setattr(module, name, fresh)
    yield
    for module, name, original in originals:
        setattr(module, name, original)
