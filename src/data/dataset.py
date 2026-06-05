"""Dataset and DataLoader utilities for accelerometer time-series arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.signal import butter, sosfiltfilt
except ImportError:  # pragma: no cover - exercised only when scipy is unavailable.
    butter = None
    sosfiltfilt = None


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


def make_stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Create deterministic stratified train/val/test indices by label."""
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        label_indices = rng.permutation(label_indices)
        num_label_samples = len(label_indices)
        train_count = int(num_label_samples * train_ratio)
        val_count = int(num_label_samples * val_ratio)
        if val_ratio > 0 and val_count == 0 and num_label_samples - train_count >= 2:
            val_count = 1
        if train_count + val_count >= num_label_samples and num_label_samples >= 3:
            train_count = max(1, num_label_samples - val_count - 1)
        train_end = train_count
        val_end = train_end + val_count
        train_parts.append(label_indices[:train_end])
        val_parts.append(label_indices[train_end:val_end])
        test_parts.append(label_indices[val_end:])

    return SplitIndices(
        train=rng.permutation(np.concatenate(train_parts)),
        val=rng.permutation(np.concatenate(val_parts)),
        test=rng.permutation(np.concatenate(test_parts)),
    )


def make_dataset_splits(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_strategy: str = "stratified",
) -> SplitIndices:
    """Create sequence-level splits using random or stratified strategy."""
    if split_strategy == "stratified":
        return make_stratified_split_indices(labels, train_ratio, val_ratio, seed)
    if split_strategy == "random":
        return make_split_indices(labels.shape[0], train_ratio, val_ratio, seed)
    raise ValueError("split_strategy must be one of: stratified, random")


def compute_train_stats(x: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from train samples only."""
    train_x = x[train_indices]
    mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def compute_filtered_train_stats(
    x: np.ndarray,
    train_indices: np.ndarray,
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train statistics after bandpass filtering."""
    total_sum: np.ndarray | None = None
    total_sq_sum: np.ndarray | None = None
    total_count = 0
    for idx in train_indices:
        filtered = apply_bandpass_filter(x[idx], sampling_rate, lowcut, highcut, filter_order)
        sample_sum = filtered.sum(axis=0, keepdims=True)
        sample_sq_sum = np.square(filtered).sum(axis=0, keepdims=True)
        total_sum = sample_sum if total_sum is None else total_sum + sample_sum
        total_sq_sum = sample_sq_sum if total_sq_sum is None else total_sq_sum + sample_sq_sum
        total_count += filtered.shape[0]
    if total_sum is None or total_sq_sum is None or total_count == 0:
        raise ValueError("Cannot compute train statistics from an empty split")
    mean = (total_sum / total_count)[None, ...].astype(np.float32)
    variance = (total_sq_sum / total_count) - np.square(total_sum / total_count)
    std = np.sqrt(np.maximum(variance, 1e-6))[None, ...].astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def make_window_starts(length: int, window_length: int, hop_length: int) -> list[int]:
    """Create sliding-window start indices."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if window_length >= length:
        return [0]
    starts = list(range(0, length - window_length + 1, hop_length))
    last_start = length - window_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def apply_bandpass_filter(
    x: np.ndarray,
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filtering along time."""
    if butter is None or sosfiltfilt is None:
        raise ImportError("scipy is required for bandpass filtering. Install scipy or set bandpass_filter: false")
    nyquist = sampling_rate * 0.5
    low = lowcut / nyquist
    high = highcut / nyquist
    if not 0 < low < high < 1:
        raise ValueError("Bandpass cutoffs must satisfy 0 < lowcut < highcut < sampling_rate / 2")
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


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
        window_mode: str = "crop",
        hop_length: int | None = None,
        taper: str = "none",
        bandpass_filter: bool = False,
        sampling_rate: float = 100.0,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        filter_order: int = 4,
        split_strategy: str = "stratified",
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        data_path = Path(data_dir)
        if source_x is None or source_y is None:
            x, y = load_array_pair(data_path, input_file, label_file, input_layout)
        else:
            x, y = source_x, source_y

        splits = make_dataset_splits(y, train_ratio, val_ratio, seed, split_strategy)
        split_indices = getattr(splits, split)
        if mean is None or std is None:
            mean, std = compute_train_stats(x, splits.train)

        self.x = x[split_indices]
        self.y = y[split_indices]
        self.items: list[tuple[int, int]] | None = None
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
        self.window_mode = window_mode
        self.hop_length = hop_length
        self.taper = taper
        self.bandpass_filter = bandpass_filter
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order

        if normalization not in {"train", "sample", "none"}:
            raise ValueError("normalization must be one of: train, sample, none")
        if window_mode not in {"crop", "sliding"}:
            raise ValueError("window_mode must be one of: crop, sliding")
        if crop_mode not in {"none", "center", "random", "random_train_center_eval"}:
            raise ValueError("crop_mode must be one of: none, center, random, random_train_center_eval")
        if transform not in {"raw", "diff", "raw_diff"}:
            raise ValueError("transform must be one of: raw, diff, raw_diff")
        if taper not in {"none", "hann"}:
            raise ValueError("taper must be one of: none, hann")
        if window_mode == "sliding":
            if window_length is None:
                raise ValueError("window_length is required when window_mode='sliding'")
            hop = hop_length or window_length
            starts = make_window_starts(x.shape[1], window_length, hop)
            self.items = [(sample_idx, start) for sample_idx in range(len(self.y)) for start in starts]

    def __len__(self) -> int:
        if self.items is not None:
            return len(self.items)
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.items is not None:
            sample_idx, start = self.items[idx]
            x = np.asarray(self.x[sample_idx], dtype=np.float32)
            x = self._preprocess_sliding_window(x, start)
            y = self.y[sample_idx]
        else:
            x = np.asarray(self.x[idx], dtype=np.float32)
            y = self.y[idx]
        if self.items is None and self.split != "train" and self.eval_num_crops > 1:
            x = self._preprocess_multicrop(x)
        elif self.items is None:
            x = self._preprocess(x)
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long)

    def _preprocess_sliding_window(self, x: np.ndarray, start: int) -> np.ndarray:
        """Apply fixed sliding-window preprocessing."""
        if self.temporal_stride > 1:
            x = x[:: self.temporal_stride]
        if self.window_length is None:
            raise ValueError("window_length is required for sliding window preprocessing")
        x = x[start : start + self.window_length]
        return self._apply_transform_and_normalization(x)

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
        if self.bandpass_filter:
            x = apply_bandpass_filter(x, self.sampling_rate, self.lowcut, self.highcut, self.filter_order)
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
        if self.taper == "hann":
            x = x * np.hanning(x.shape[0]).astype(np.float32)[:, None]
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
    window_mode: str = "crop",
    hop_length: int | None = None,
    taper: str = "none",
    bandpass_filter: bool = False,
    sampling_rate: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    filter_order: int = 4,
    split_strategy: str = "stratified",
    jitter_std: float = 0.01,
    scaling_std: float = 0.1,
    time_mask_ratio: float = 0.05,
    loader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create train/val/test DataLoaders sharing train-set normalization."""
    data_path = Path(data_dir)
    x, y = load_array_pair(data_path, input_file, label_file, input_layout, mmap_mode="r")
    splits = make_dataset_splits(y, train_ratio, val_ratio, seed, split_strategy)
    if normalization == "train":
        if bandpass_filter:
            mean, std = compute_filtered_train_stats(
                x,
                splits.train,
                sampling_rate=sampling_rate,
                lowcut=lowcut,
                highcut=highcut,
                filter_order=filter_order,
            )
        else:
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
            jitter_std=jitter_std,
            scaling_std=scaling_std,
            time_mask_ratio=time_mask_ratio,
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
            window_mode=window_mode,
            hop_length=hop_length,
            taper=taper,
            bandpass_filter=bandpass_filter,
            sampling_rate=sampling_rate,
            lowcut=lowcut,
            highcut=highcut,
            filter_order=filter_order,
            split_strategy=split_strategy,
        )
        for split in ("train", "val", "test")
    }
    return (
        DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs),
        DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs),
        DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs),
    )
