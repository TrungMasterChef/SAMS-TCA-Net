"""SAMS-TCA-Net for time-series accelerometer classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


InputLayout = Literal["btc", "bct"]


class SensorAxisAttention(nn.Module):
    """Attention over raw sensor axes before temporal feature extraction."""

    def __init__(self, num_channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, num_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel-wise weights to input shaped [B, C, T]."""
        context = x.mean(dim=-1)
        weights = self.mlp(context).unsqueeze(-1)
        return x * weights


class TemporalChannelAttention1D(nn.Module):
    """CBAM-style channel and temporal attention for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 8, temporal_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        padding = temporal_kernel // 2
        self.temporal_conv = nn.Conv1d(
            2,
            1,
            kernel_size=temporal_kernel,
            padding=padding,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel attention followed by temporal attention."""
        avg_pool = F.adaptive_avg_pool1d(x, 1)
        max_pool = F.adaptive_max_pool1d(x, 1)
        channel_attn = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_attn

        avg_time = x.mean(dim=1, keepdim=True)
        max_time, _ = x.max(dim=1, keepdim=True)
        temporal_attn = torch.sigmoid(self.temporal_conv(torch.cat([avg_time, max_time], dim=1)))
        return x * temporal_attn


class ResidualInceptionAttentionBlock(nn.Module):
    """Multi-scale residual Conv1D block with scale and temporal-channel attention."""

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, ...] = (3, 5, 9, 15),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernels
            ]
        )
        self.scale_attention = nn.Linear(channels, len(kernels))
        self.tca = TemporalChannelAttention1D(channels)
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Return residual-enhanced features shaped [B, C, T]."""
        branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=1)
        context = branch_outputs.mean(dim=-1).mean(dim=1)
        weights = torch.softmax(self.scale_attention(context), dim=-1)
        mixed = (branch_outputs * weights[:, :, None, None]).sum(dim=1)
        mixed = self.tca(mixed)
        mixed = self.proj(mixed)
        return self.activation(x + mixed)


class ClassAwareAttentionPooling(nn.Module):
    """Class-aware temporal pooling that produces logits directly."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.attention = nn.Conv1d(channels, num_classes, kernel_size=1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Pool temporal features into class logits shaped [B, num_classes]."""
        attn = torch.softmax(self.attention(x), dim=-1)
        pooled = torch.einsum("bkt,bct->bkc", attn, x)
        class_weights = self.classifier.weight.unsqueeze(0)
        logits = (pooled * class_weights).sum(dim=-1) + self.classifier.bias.unsqueeze(0)
        return logits


@dataclass(frozen=True)
class SAMSTCANetConfig:
    """Configuration values for SAMSTCANet."""

    num_channels: int
    num_classes: int
    hidden_channels: int = 64
    num_blocks: int = 4
    dropout: float = 0.1
    input_layout: InputLayout = "btc"


class SAMSTCANet(nn.Module):
    """Sensor Axis Multi-Scale Temporal-Channel Attention Network.

    Args:
        num_channels: Number of input sensor channels.
        num_classes: Number of output classes.
        hidden_channels: Internal Conv1D feature width.
        num_blocks: Number of residual inception attention blocks.
        dropout: Dropout probability inside residual blocks.
        input_layout: Expected input layout, either ``"btc"`` for [B, T, C]
            or ``"bct"`` for [B, C, T].
    """

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
        if input_layout not in {"btc", "bct"}:
            raise ValueError("input_layout must be 'btc' or 'bct'")

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_layout = input_layout

        self.sensor_attention = SensorAxisAttention(num_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualInceptionAttentionBlock(
                    channels=hidden_channels,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.pooling = ClassAwareAttentionPooling(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits without softmax.

        Accepts input shaped [B, T, C] when ``input_layout="btc"`` or [B, C, T]
        when ``input_layout="bct"``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        if self.input_layout == "btc":
            if x.shape[-1] != self.num_channels:
                raise ValueError(f"Expected last dim C={self.num_channels}, got {x.shape[-1]}")
            x = x.transpose(1, 2)
        elif x.shape[1] != self.num_channels:
            raise ValueError(f"Expected channel dim C={self.num_channels}, got {x.shape[1]}")

        x = self.sensor_attention(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.pooling(x)
