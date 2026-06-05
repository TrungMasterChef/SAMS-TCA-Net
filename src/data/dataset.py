"""Dataset and DataLoader utilities for accelerometer time-series arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SplitIndices:
    """Train/validation/test index splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split_indices(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Create deterministic shuffled train/val/test indices."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    return SplitIndices(
        train=indices[:train_end],
        val=indices[train_end:val_end],
        test=indices[val_end:],
    )


def compute_train_stats(x: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from train samples only."""
    train_x = x[train_indices]
    mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


class Z24AccelerationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset for ``X.npy`` [N, T, C] and ``y.npy`` [N] accelerometer data."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        augment: bool = False,
        jitter_std: float = 0.01,
        scaling_std: float = 0.1,
        time_mask_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        data_path = Path(data_dir)
        x_path = data_path / "X.npy"
        y_path = data_path / "y.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Expected {x_path} and {y_path}")

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        if x.ndim != 3:
            raise ValueError(f"X.npy must have shape [N, T, C], got {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError(f"y.npy must have shape [N], got {y.shape} for X shape {x.shape}")

        splits = make_split_indices(x.shape[0], train_ratio, val_ratio, seed)
        split_indices = getattr(splits, split)
        if mean is None or std is None:
            mean, std = compute_train_stats(x, splits.train)

        self.x = x[split_indices]
        self.y = y[split_indices]
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment and split == "train"
        self.jitter_std = jitter_std
        self.scaling_std = scaling_std
        self.time_mask_ratio = time_mask_ratio

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = (self.x[idx] - self.mean.squeeze(0)) / self.std.squeeze(0)
        if self.augment:
            x = self._augment(x)
        y = self.y[idx]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply jitter, scaling, and contiguous time masking."""
        if self.jitter_std > 0:
            x = x + np.random.normal(0.0, self.jitter_std, size=x.shape).astype(np.float32)
        if self.scaling_std > 0:
            scale = np.random.normal(1.0, self.scaling_std, size=(1, x.shape[-1])).astype(np.float32)
            x = x * scale
        if self.time_mask_ratio > 0:
            mask_len = int(round(x.shape[0] * self.time_mask_ratio))
            if mask_len > 0 and mask_len < x.shape[0]:
                start = np.random.randint(0, x.shape[0] - mask_len + 1)
                x = x.copy()
                x[start : start + mask_len, :] = 0.0
        return x


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    augment: bool = True,
    loader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create train/val/test DataLoaders sharing train-set normalization."""
    data_path = Path(data_dir)
    x = np.load(data_path / "X.npy").astype(np.float32)
    splits = make_split_indices(x.shape[0], train_ratio, val_ratio, seed)
    mean, std = compute_train_stats(x, splits.train)
    kwargs = loader_kwargs or {}

    datasets = {
        split: Z24AccelerationDataset(
            data_dir=data_path,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            mean=mean,
            std=std,
            augment=augment,
        )
        for split in ("train", "val", "test")
    }
    return (
        DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs),
        DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs),
        DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs),
    )
