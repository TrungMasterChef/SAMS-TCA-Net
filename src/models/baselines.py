"""Baseline 1D neural networks for accelerometer classification."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
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


class PatchTST(nn.Module):
    """PatchTST: patching + channel-independent Transformer (Nie et al., ICLR 2023)."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        num_heads: int = 4,
        patch_len: int = 32,
        patch_stride: int = 16,
        mlp_ratio: float = 2.0,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.embed = nn.Linear(patch_len, hidden_channels)
        position = torch.arange(max_patches, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_channels, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_channels))
        encoding = torch.zeros(max_patches, hidden_channels)
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
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_channels * hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        if x.shape[-1] < self.patch_len:
            x = F.pad(x, (0, self.patch_len - x.shape[-1]))
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        batch_size, channels, num_patches, _ = patches.shape
        z = self.embed(patches).reshape(batch_size * channels, num_patches, -1)
        z = z + self.positional_encoding[:, :num_patches]
        z = self.norm(self.encoder(z)).mean(dim=1)
        z = self.dropout(z.reshape(batch_size, channels * z.shape[-1]))
        return self.head(z)


class _MixerBlock(nn.Module):
    """TSMixer block: token (time) mixing then feature mixing, both residual."""

    def __init__(self, num_tokens: int, dim: int, dropout: float = 0.1, expand: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.time_mlp = nn.Sequential(nn.Linear(num_tokens, num_tokens), nn.GELU(), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(dim, dim * expand), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * expand, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Mix ``[B, tokens, dim]`` over time and features."""
        x = x + self.time_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.feat_mlp(self.norm2(x))
        return x


class TSMixer(nn.Module):
    """TSMixer: an all-MLP time-series model (Chen et al., 2023).

    A convolutional patch stem provides a fixed-length token grid, then stacked
    mixer blocks alternate time-mixing and feature-mixing MLPs.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        num_tokens: int = 96,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.blocks = nn.ModuleList(_MixerBlock(num_tokens, hidden_channels, dropout) for _ in range(num_blocks))
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        h = self.pool(self.stem(x)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.classifier(self.dropout(self.norm(h).mean(dim=1)))


class _MambaBlock(nn.Module):
    """Minimal selective state-space (Mamba) block in pure PyTorch."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_inner = expand * dim
        self.d_state = d_state
        self.dt_rank = max(1, dim // 16)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a selective scan to a ``[B, L, dim]`` sequence."""
        batch_size, length, _ = x.shape
        projected, gate = self.in_proj(x).chunk(2, dim=-1)
        projected = self.conv1d(projected.transpose(1, 2))[..., :length].transpose(1, 2)
        projected = F.silu(projected)

        dt, b_mat, c_mat = torch.split(
            self.x_proj(projected), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        a_mat = -torch.exp(self.A_log)
        delta_a = torch.exp(dt.unsqueeze(-1) * a_mat)
        delta_bx = dt.unsqueeze(-1) * b_mat.unsqueeze(2) * projected.unsqueeze(-1)

        state = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(length):
            state = delta_a[:, step] * state + delta_bx[:, step]
            outputs.append((state * c_mat[:, step].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(outputs, dim=1) + projected * self.D
        return self.out_proj(y * F.silu(gate))


class MambaSL(nn.Module):
    """Mamba (selective state-space) sequence-learning classifier."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_layout = input_layout
        self.tokenizer = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(_MambaBlock(hidden_channels, d_state=d_state) for _ in range(num_blocks))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_blocks))
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits shaped [B, num_classes]."""
        x = _to_bct(x, self.num_channels, self.input_layout)
        tokens = self.tokenizer(x).transpose(1, 2)
        for block, norm in zip(self.blocks, self.norms):
            tokens = tokens + block(norm(tokens))
        return self.classifier(self.dropout(self.norm(tokens).mean(dim=1)))


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
    """Plain InceptionTime baseline (no attention or class-aware pooling)."""

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
