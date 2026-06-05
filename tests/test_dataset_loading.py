import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import Z24AccelerationDataset, create_dataloaders, make_stratified_split_indices


def test_dataset_loads_nct_and_returns_tc(tmp_path: Path) -> None:
    x = np.random.randn(20, 3, 16).astype(np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    dataset = Z24AccelerationDataset(
        tmp_path,
        split="train",
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        augment=False,
    )

    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (16, 3)
    assert sample_x.dtype == torch.float32
    assert sample_y.dtype == torch.long


def test_dataloader_batches_are_btc(tmp_path: Path) -> None:
    x = np.random.randn(20, 3, 16).astype(np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    train_loader, _, _ = create_dataloaders(
        tmp_path,
        batch_size=5,
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        augment=False,
    )
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (5, 16, 3)
    assert batch_y.shape == (5,)


def test_harder_preprocessing_returns_windowed_diff_sample_normalized_data(tmp_path: Path) -> None:
    x = np.random.randn(20, 3, 32).astype(np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    train_loader, val_loader, _ = create_dataloaders(
        tmp_path,
        batch_size=5,
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        normalization="sample",
        window_length=12,
        crop_mode="random_train_center_eval",
        temporal_stride=1,
        transform="diff",
        augment=False,
    )
    train_x, _ = next(iter(train_loader))
    val_x, _ = next(iter(val_loader))
    assert train_x.shape == (5, 12, 3)
    assert val_x.shape == (4, 12, 3)
    assert torch.isfinite(train_x).all()
    assert torch.isfinite(val_x).all()


def test_raw_diff_transform_doubles_channel_count(tmp_path: Path) -> None:
    x = np.random.randn(20, 3, 32).astype(np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    train_loader, _, _ = create_dataloaders(
        tmp_path,
        batch_size=5,
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        normalization="sample",
        window_length=12,
        crop_mode="random_train_center_eval",
        transform="raw_diff",
        augment=False,
    )
    batch_x, _ = next(iter(train_loader))
    assert batch_x.shape == (5, 12, 6)
    assert torch.isfinite(batch_x).all()


def test_eval_multicrop_batches_are_bktc(tmp_path: Path) -> None:
    x = np.random.randn(20, 3, 32).astype(np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    _, val_loader, _ = create_dataloaders(
        tmp_path,
        batch_size=3,
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        normalization="sample",
        window_length=12,
        crop_mode="random_train_center_eval",
        transform="raw_diff",
        eval_num_crops=4,
        augment=False,
    )
    batch_x, batch_y = next(iter(val_loader))
    assert batch_x.shape == (3, 4, 12, 6)
    assert batch_y.shape == (3,)
    assert torch.isfinite(batch_x).all()


def test_sliding_window_preprocessing_repeats_labels_and_returns_windows(tmp_path: Path) -> None:
    x = np.random.randn(20, 3, 32).astype(np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    train_loader, _, _ = create_dataloaders(
        tmp_path,
        batch_size=4,
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        normalization="train",
        window_mode="sliding",
        window_length=12,
        hop_length=6,
        transform="raw",
        taper="hann",
        augment=False,
    )
    batch_x, batch_y = next(iter(train_loader))
    assert len(train_loader.dataset) == 12 * 5
    assert batch_x.shape == (4, 12, 3)
    assert batch_y.shape == (4,)
    assert torch.isfinite(batch_x).all()


def test_stratified_split_balances_each_class() -> None:
    labels = np.repeat(np.arange(4), 10)
    splits = make_stratified_split_indices(labels, train_ratio=0.6, val_ratio=0.2, seed=42)
    assert np.bincount(labels[splits.train], minlength=4).tolist() == [6, 6, 6, 6]
    assert np.bincount(labels[splits.val], minlength=4).tolist() == [2, 2, 2, 2]
    assert np.bincount(labels[splits.test], minlength=4).tolist() == [2, 2, 2, 2]


def test_channel_mask_augmentation_masks_some_channels(tmp_path: Path) -> None:
    x = np.ones((20, 3, 32), dtype=np.float32)
    y = np.arange(20, dtype=np.int64) % 4
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)

    dataset = Z24AccelerationDataset(
        tmp_path,
        split="train",
        input_file="inputs.npy",
        label_file="labels.npy",
        input_layout="nct",
        normalization="none",
        window_length=16,
        crop_mode="center",
        augment=True,
        jitter_std=0.0,
        scaling_std=0.0,
        time_mask_ratio=0.0,
        channel_mask_ratio=1 / 3,
    )
    sample_x, _ = dataset[0]
    assert sample_x.shape == (16, 3)
    assert (sample_x.sum(dim=0) == 0).sum().item() == 1
