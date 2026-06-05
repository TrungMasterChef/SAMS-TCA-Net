"""Baseline 1D neural networks for accelerometer classification."""

from __future__ import annotations

import math
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


def _to_btc(x: Tensor, num_channels: int, input_layout: InputLayout) -> Tensor:
    """Convert [B, T, C] or [B, C, T] input to [B, T, C]."""
    return _to_bct(x, num_channels, input_layout).transpose(1, 2)


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


class MLP1D(nn.Module):
    """Multilayer perceptron baseline over pooled per-channel features."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        pool_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        in_features = num_channels * pool_size
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        x = self.pool(x).flatten(1)
        return self.mlp(x)


class LSTM1D(nn.Module):
    """Bidirectional LSTM baseline with mean+last temporal read-out."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_channels,
            num_layers=num_blocks,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_blocks > 1 else 0.0,
        )
        output_dim = hidden_channels * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_btc(x, self.num_channels, self.input_layout)
        sequence, _ = self.lstm(x)
        pooled = sequence.mean(dim=1)
        return self.classifier(self.dropout(pooled))


class _TemporalBlock(nn.Module):
    """Dilated residual Conv1D block for the TCN baseline."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the dilated residual transformation."""
        return self.activation(self.conv(x) + self.shortcut(x))


class TCN1D(nn.Module):
    """Temporal Convolutional Network baseline with exponentially growing dilation."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        blocks: list[nn.Module] = []
        in_channels = num_channels
        for index in range(num_blocks):
            blocks.append(
                _TemporalBlock(in_channels, hidden_channels, kernel_size, dilation=2 ** index, dropout=dropout)
            )
            in_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        x = self.network(x)
        return self.classifier(x.mean(dim=-1))


class TransformerClassifier(nn.Module):
    """Transformer-encoder baseline over a strided-conv token sequence."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_len: int = 8192,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.tokenizer = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_channels, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_channels))
        encoding = torch.zeros(max_len, hidden_channels)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", encoding.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=int(hidden_channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        tokens = self.tokenizer(x).transpose(1, 2)
        tokens = tokens + self.positional_encoding[:, : tokens.size(1)]
        encoded = self.norm(self.encoder(tokens))
        return self.classifier(encoded.mean(dim=1))


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
