from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ModelEMA:
    """Maintain one directly deployable EMA copy for ECD evaluation."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between 0 and 1")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self._parameters = tuple(self.model.parameters())
        self._source_parameters = tuple(model.parameters())
        self._buffers = tuple(self.model.buffers())
        self._source_buffers = tuple(model.buffers())
        if len(self._parameters) != len(self._source_parameters):
            raise RuntimeError("EMA parameter structure does not match the source model")
        if len(self._buffers) != len(self._source_buffers):
            raise RuntimeError("EMA buffer structure does not match the source model")

    @torch.no_grad()
    def update(self) -> None:
        if self._parameters:
            torch._foreach_lerp_(self._parameters, self._source_parameters, 1.0 - self.decay)
        for averaged, source in zip(self._buffers, self._source_buffers):
            averaged.copy_(source)
