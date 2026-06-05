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


def load_array_pair(
    data_dir: str | Path,
    input_file: str = "X.npy",
    label_file: str = "y.npy",
    input_layout: str = "ntc",
    mmap_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load arrays and convert inputs to canonical [N, T, C] layout."""
    data_path = Path(data_dir)
    x_path = data_path / input_file
    y_path = data_path / label_file
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected {x_path} and {y_path}")

    x = np.load(x_path, mmap_mode=mmap_mode)
    y = np.load(y_path, mmap_mode=mmap_mode)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if y.dtype != np.int64:
        y = y.astype(np.int64)
    if x.ndim != 3:
        raise ValueError(f"{input_file} must have 3 dimensions, got {x.shape}")
    if input_layout == "nct":
        x = np.transpose(x, (0, 2, 1))
    elif input_layout != "ntc":
        raise ValueError("input_layout must be 'ntc' for [N, T, C] or 'nct' for [N, C, T]")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"{label_file} must have shape [N], got {y.shape} for X shape {x.shape}")
    return x, y


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
        input_file: str = "X.npy",
        label_file: str = "y.npy",
        input_layout: str = "ntc",
        source_x: np.ndarray | None = None,
        source_y: np.ndarray | None = None,
        normalization: str = "train",
        window_length: int | None = None,
        crop_mode: str = "none",
        temporal_stride: int = 1,
        transform: str = "raw",
        eval_num_crops: int = 1,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        data_path = Path(data_dir)
        if source_x is None or source_y is None:
            x, y = load_array_pair(data_path, input_file, label_file, input_layout)
        else:
            x, y = source_x, source_y

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
        self.split = split
        self.normalization = normalization
        self.window_length = window_length
        self.crop_mode = crop_mode
        self.temporal_stride = max(1, temporal_stride)
        self.transform = transform
        self.eval_num_crops = max(1, eval_num_crops)

        if normalization not in {"train", "sample", "none"}:
            raise ValueError("normalization must be one of: train, sample, none")
        if crop_mode not in {"none", "center", "random", "random_train_center_eval"}:
            raise ValueError("crop_mode must be one of: none, center, random, random_train_center_eval")
        if transform not in {"raw", "diff", "raw_diff"}:
            raise ValueError("transform must be one of: raw, diff, raw_diff")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = np.asarray(self.x[idx], dtype=np.float32)
        if self.split != "train" and self.eval_num_crops > 1:
            x = self._preprocess_multicrop(x)
        else:
            x = self._preprocess(x)
        if self.augment:
            x = self._augment(x)
        y = self.y[idx]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long)

    def _preprocess_multicrop(self, x: np.ndarray) -> np.ndarray:
        """Return deterministic evaluation crops shaped [K, T, C]."""
        if self.temporal_stride > 1:
            x = x[:: self.temporal_stride]
        if self.window_length is None or self.window_length <= 0 or self.window_length >= x.shape[0]:
            return self._apply_transform_and_normalization(x)[None, ...]

        max_start = x.shape[0] - self.window_length
        starts = np.linspace(0, max_start, num=self.eval_num_crops).round().astype(np.int64)
        crops = [self._apply_transform_and_normalization(x[start : start + self.window_length]) for start in starts]
        return np.stack(crops, axis=0)

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply deterministic preprocessing before optional augmentation."""
        if self.temporal_stride > 1:
            x = x[:: self.temporal_stride]
        x = self._crop(x)
        return self._apply_transform_and_normalization(x)

    def _apply_transform_and_normalization(self, x: np.ndarray) -> np.ndarray:
        """Apply signal transform and normalization to one temporal window."""
        if self.transform == "diff":
            x = np.diff(x, axis=0, prepend=x[:1])
        elif self.transform == "raw_diff":
            diff = np.diff(x, axis=0, prepend=x[:1])
            x = np.concatenate([x, diff], axis=-1)
        if self.normalization == "train":
            if x.shape[-1] != self.mean.shape[-1]:
                raise ValueError("train normalization is incompatible with transforms that change channel count")
            x = (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        elif self.normalization == "sample":
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            x = (x - mean) / std
        return x.astype(np.float32)

    def _crop(self, x: np.ndarray) -> np.ndarray:
        """Crop a temporal window according to the configured crop mode."""
        if self.window_length is None or self.window_length <= 0 or self.window_length >= x.shape[0]:
            return x

        if self.crop_mode == "none":
            return x
        if self.crop_mode == "random" or (
            self.crop_mode == "random_train_center_eval" and self.split == "train"
        ):
            start = np.random.randint(0, x.shape[0] - self.window_length + 1)
        else:
            start = (x.shape[0] - self.window_length) // 2
        return x[start : start + self.window_length]

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
    input_file: str = "X.npy",
    label_file: str = "y.npy",
    input_layout: str = "ntc",
    normalization: str = "train",
    window_length: int | None = None,
    crop_mode: str = "none",
    temporal_stride: int = 1,
    transform: str = "raw",
    eval_num_crops: int = 1,
    loader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create train/val/test DataLoaders sharing train-set normalization."""
    data_path = Path(data_dir)
    x, y = load_array_pair(data_path, input_file, label_file, input_layout, mmap_mode="r")
    splits = make_split_indices(x.shape[0], train_ratio, val_ratio, seed)
    if normalization == "train":
        mean, std = compute_train_stats(x, splits.train)
    else:
        mean = np.zeros((1, 1, x.shape[-1]), dtype=np.float32)
        std = np.ones((1, 1, x.shape[-1]), dtype=np.float32)
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
            input_file=input_file,
            label_file=label_file,
            input_layout=input_layout,
            source_x=x,
            source_y=y,
            normalization=normalization,
            window_length=window_length,
            crop_mode=crop_mode,
            temporal_stride=temporal_stride,
            transform=transform,
            eval_num_crops=eval_num_crops,
        )
        for split in ("train", "val", "test")
    }
    return (
        DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs),
        DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs),
        DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs),
    )
