"""Baseline 1D neural networks for accelerometer classification."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


InputLayout = Literal["btc", "bct"]


def _to_bct(x: Tensor, num_channels: int, input_layout: InputLayout) -> Tensor:
    """Convert [B, T, C] or [B, C, T] input to [B, C, T]."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")
    if input_layout == "btc":
        if x.shape[-1] != num_channels:
            raise ValueError(f"Expected last dim C={num_channels}, got {x.shape[-1]}")
        return x.transpose(1, 2)
    if input_layout == "bct":
        if x.shape[1] != num_channels:
            raise ValueError(f"Expected channel dim C={num_channels}, got {x.shape[1]}")
        return x
    raise ValueError("input_layout must be 'btc' or 'bct'")


class ConvBNReLU(nn.Sequential):
    """Conv1D, BatchNorm1D, and ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleCNN1D(nn.Module):
    """Small Conv1D baseline with global average pooling."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.features = nn.Sequential(
            ConvBNReLU(num_channels, hidden_channels, kernel_size=7),
            nn.MaxPool1d(kernel_size=2),
            ConvBNReLU(hidden_channels, hidden_channels * 2, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            ConvBNReLU(hidden_channels * 2, hidden_channels * 2, kernel_size=3),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        x = self.features(x)
        x = x.mean(dim=-1)
        return self.classifier(self.dropout(x))


class FCN1D(nn.Module):
    """Fully convolutional network baseline for time-series classification."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.features = nn.Sequential(
            ConvBNReLU(num_channels, hidden_channels, kernel_size=8),
            ConvBNReLU(hidden_channels, hidden_channels * 2, kernel_size=5),
            ConvBNReLU(hidden_channels * 2, hidden_channels, kernel_size=3),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        x = self.features(x)
        return self.classifier(x.mean(dim=-1))


class ResNetBlock1D(nn.Module):
    """Basic residual block for 1D time-series features."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=kernel_size),
            ConvBNReLU(out_channels, out_channels, kernel_size=kernel_size),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual transformation."""
        return self.activation(self.layers(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """ResNet-style 1D baseline with global average pooling."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        input_layout: InputLayout = "btc",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        blocks: list[nn.Module] = []
        in_channels = num_channels
        for idx in range(num_blocks):
            out_channels = hidden_channels * min(2 ** idx, 4)
            blocks.append(ResNetBlock1D(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        x = self.features(x)
        return self.classifier(x.mean(dim=-1))


class InceptionModule1D(nn.Module):
    """Plain InceptionTime module without attention mechanisms."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int = 32,
        kernels: tuple[int, ...] = (3, 5, 9, 15),
    ) -> None:
        super().__init__()
        use_bottleneck = in_channels > 1
        branch_in = bottleneck_channels if use_bottleneck else in_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if use_bottleneck
            else nn.Identity()
        )
        branch_channels = out_channels // (len(kernels) + 1)
        pool_channels = out_channels - branch_channels * len(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(branch_in, branch_channels, kernel_size=k, padding=k // 2, bias=False)
                for k in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_channels, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Concatenate plain multi-scale Conv1D branches."""
        bottleneck = self.bottleneck(x)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(x))
        return self.activation(self.norm(torch.cat(branches, dim=1)))


class InceptionTimeBaseline(nn.Module):
    """InceptionTime baseline without SAMS-TCA attention or class-aware pooling."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        blocks: list[nn.Module] = []
        in_channels = num_channels
        for _ in range(num_blocks):
            blocks.append(InceptionModule1D(in_channels, hidden_channels))
            in_channels = hidden_channels
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        x = self.features(x)
        x = x.mean(dim=-1)
        return self.classifier(self.dropout(x))
