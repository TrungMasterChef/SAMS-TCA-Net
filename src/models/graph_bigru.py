"""AGB-Net: an Adaptive-Graph + BiGRU hybrid with dual spatio-temporal attention.

The model treats the ``C`` accelerometer channels as nodes of a sensor graph and
combines three complementary inductive biases for structural-health monitoring:

* a **shared node temporal encoder** (weight-shared 1D convolutions) that lifts
  each per-sensor signal into a downsampled temporal feature sequence;
* an **adaptive graph convolution** that mixes information across sensors through
  a *learned* adjacency matrix, so the model can discover which sensors are
  coupled without needing physical coordinates (damage changes inter-sensor
  correlations, which a fixed graph cannot capture); and
* a **bidirectional GRU** that models forward/backward temporal dynamics.

Two attention read-outs make the representation compact and interpretable: a
**spatial attention** over sensors and a **temporal attention** over time. Every
component is an ablation flag, so the network can be reduced to a vanilla
GCN-BiGRU (fixed graph, mean pooling, unidirectional) for comparison.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .sams_tca_net import make_group_norm

InputLayout = Literal["btc", "bct"]


class NodeTemporalEncoder(nn.Module):
    """Weight-shared 1D-conv encoder applied independently to every sensor node."""

    def __init__(self, hidden_channels: int, conv_blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
                make_group_norm(hidden_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            )
            for _ in range(conv_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``[B*N, 1, T]`` per-node signals to ``[B*N, hidden, T']`` features."""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class AdaptiveGraphConv(nn.Module):
    """Chebyshev-style graph convolution over the node dimension."""

    def __init__(self, in_features: int, out_features: int, order: int = 2) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("graph_order must be >= 1")
        self.order = order
        self.linear = nn.Linear((order + 1) * in_features, out_features)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """Aggregate ``[B, T, N, F]`` features over ``order`` hops of ``adjacency``."""
        supports = [features]
        propagated = features
        for _ in range(self.order):
            propagated = torch.einsum("nm,btmf->btnf", adjacency, propagated)
            supports.append(propagated)
        return self.linear(torch.cat(supports, dim=-1))


class SpatialAttentionPool(nn.Module):
    """Attention pooling over sensor nodes."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.score = nn.Linear(features, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Pool ``[B, T, N, F]`` into ``[B, T, F]`` with per-node attention weights."""
        weights = torch.softmax(self.score(features), dim=2)
        return (weights * features).sum(dim=2)


class TemporalAttentionPool(nn.Module):
    """Additive attention pooling over the time dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, sequence: Tensor) -> Tensor:
        """Pool ``[B, T, D]`` into ``[B, D]`` with per-time-step attention weights."""
        weights = torch.softmax(self.score(torch.tanh(self.projection(sequence))), dim=1)
        return (weights * sequence).sum(dim=1)


class GraphBiGRUNet(nn.Module):
    """Adaptive-Graph BiGRU with spatio-temporal attention (AGB-Net).

    Args:
        num_channels: Number of input sensor channels (graph nodes).
        num_classes: Number of output classes.
        hidden_channels: Node feature width and GRU hidden size (per direction).
        num_blocks: Number of stacked GRU layers.
        conv_blocks: Number of downsampling blocks in the node temporal encoder.
        graph_order: Number of graph-propagation hops (Chebyshev order).
        node_embedding_dim: Dimension of the learnable node embeddings.
        dropout: Dropout used in the encoder, GRU, and classifier.
        input_layout: ``"btc"`` for ``[B, T, C]`` or ``"bct"`` for ``[B, C, T]``.
        use_graph: Enable graph convolution (else a node-wise linear, no mixing).
        use_adaptive_graph: Learn the adjacency (else a fixed uniform graph).
        use_spatial_attention: Pool sensors with attention (else mean).
        use_temporal_attention: Pool time with attention (else mean).
        bidirectional: Use a bidirectional GRU (else a unidirectional GRU).
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 1,
        conv_blocks: int = 2,
        graph_order: int = 2,
        node_embedding_dim: int = 10,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        use_graph: bool = True,
        use_adaptive_graph: bool = True,
        use_spatial_attention: bool = True,
        use_temporal_attention: bool = True,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if input_layout not in {"btc", "bct"}:
            raise ValueError("input_layout must be 'btc' or 'bct'")

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_layout = input_layout
        self.use_graph = use_graph
        self.use_adaptive_graph = use_adaptive_graph
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention

        self.node_encoder = NodeTemporalEncoder(hidden_channels, conv_blocks=conv_blocks, dropout=dropout)

        if use_graph:
            if use_adaptive_graph:
                self.node_embeddings = nn.Parameter(torch.empty(num_channels, node_embedding_dim))
                nn.init.xavier_uniform_(self.node_embeddings)
            else:
                uniform = torch.full((num_channels, num_channels), 1.0 / num_channels)
                self.register_buffer("fixed_adjacency", uniform)
            self.graph_conv = AdaptiveGraphConv(hidden_channels, hidden_channels, order=graph_order)
            self.graph_norm = nn.LayerNorm(hidden_channels)
            self.graph_activation = nn.GELU()
        else:
            self.node_projection = nn.Linear(hidden_channels, hidden_channels)

        self.spatial_pool = SpatialAttentionPool(hidden_channels) if use_spatial_attention else None

        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_blocks,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_blocks > 1 else 0.0,
        )
        gru_output_dim = hidden_channels * (2 if bidirectional else 1)
        self.temporal_pool = TemporalAttentionPool(gru_output_dim) if use_temporal_attention else None

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, num_classes),
        )

    def _adjacency(self) -> Tensor:
        """Return the row-stochastic adjacency used for graph propagation."""
        if self.use_adaptive_graph:
            similarity = torch.relu(self.node_embeddings @ self.node_embeddings.transpose(0, 1))
            return torch.softmax(similarity, dim=1)
        return self.fixed_adjacency

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits (without softmax) shaped ``[B, num_classes]``."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")
        if self.input_layout == "btc":
            if x.shape[-1] != self.num_channels:
                raise ValueError(f"Expected last dim C={self.num_channels}, got {x.shape[-1]}")
            x = x.transpose(1, 2)
        elif x.shape[1] != self.num_channels:
            raise ValueError(f"Expected channel dim C={self.num_channels}, got {x.shape[1]}")

        batch_size, num_nodes, time_steps = x.shape
        encoded = self.node_encoder(x.reshape(batch_size * num_nodes, 1, time_steps))
        features = encoded.reshape(batch_size, num_nodes, encoded.size(1), encoded.size(2))
        features = features.permute(0, 3, 1, 2)  # [B, T', N, F]

        if self.use_graph:
            mixed = self.graph_conv(features, self._adjacency())
            features = self.graph_activation(self.graph_norm(mixed + features))
        else:
            features = self.node_projection(features)

        if self.spatial_pool is not None:
            sequence = self.spatial_pool(features)  # [B, T', F]
        else:
            sequence = features.mean(dim=2)

        temporal, _ = self.gru(sequence)  # [B, T', gru_output_dim]
        pooled = self.temporal_pool(temporal) if self.temporal_pool is not None else temporal.mean(dim=1)
        return self.classifier(pooled)
