"""Synthetic-data splits as surrogate tensors (standardised features + FFR targets)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vstent.data.synth import load_split
from vstent.surrogate.features import NormStats, RawInputs, prepare_inputs

RAW_KEYS = ("r", "r_ref", "length", "pa", "pv", "r_micro")


def load_raw(
    data_dir: str | Path, split: str, max_samples: int | None = None
) -> tuple[RawInputs, np.ndarray]:
    """Raw SI inputs and the solver FFR curves (B, N) of one split, read from ``.npz`` shards."""
    arrays = load_split(data_dir, split, (*RAW_KEYS, "ffr"))
    if max_samples is not None:
        arrays = {k: v[:max_samples] for k, v in arrays.items()}
    return RawInputs.from_arrays(arrays), arrays["ffr"].astype(np.float32)


class SurrogateDataset(Dataset):
    """Standardised inputs and FFR targets held as tensors.

    Usable with a ``DataLoader``; for speed, training moves the whole split to the GPU and
    draws batches with :meth:`batches`.
    """

    def __init__(self, raw: RawInputs, ffr: np.ndarray, stats: NormStats) -> None:
        xp, xg = prepare_inputs(raw, stats)
        self.x_point = torch.from_numpy(xp)
        self.x_global = torch.from_numpy(xg)
        self.ffr = torch.from_numpy(np.ascontiguousarray(ffr, dtype=np.float32))

    @classmethod
    def from_split(
        cls, data_dir: str | Path, split: str, stats: NormStats, max_samples: int | None = None
    ) -> "SurrogateDataset":
        raw, ffr = load_raw(data_dir, split, max_samples)
        return cls(raw, ffr, stats)

    def __len__(self) -> int:
        return self.ffr.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_point[i], self.x_global[i], self.ffr[i]

    def to(self, device: torch.device) -> "SurrogateDataset":
        """Move all tensors to ``device`` (in place); returns self."""
        self.x_point = self.x_point.to(device)
        self.x_global = self.x_global.to(device)
        self.ffr = self.ffr.to(device)
        return self

    def batches(
        self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (x_point, x_global, ffr) batches without per-sample Python overhead."""
        n = len(self)
        device = self.ffr.device
        order = (
            torch.randperm(n, generator=generator, device=device)
            if shuffle
            else torch.arange(n, device=device)
        )
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield self.x_point[idx], self.x_global[idx], self.ffr[idx]
