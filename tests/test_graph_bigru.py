import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import GraphBiGRUNet, build_model


def test_forward_btc_shape() -> None:
    model = GraphBiGRUNet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=1, conv_blocks=2)
    logits = model(torch.randn(4, 256, 3))
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_forward_bct_shape() -> None:
    model = GraphBiGRUNet(
        num_channels=3,
        num_classes=5,
        hidden_channels=16,
        num_blocks=2,
        conv_blocks=2,
        input_layout="bct",
    )
    logits = model(torch.randn(4, 3, 256))
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_factory_builds_graph_bigru() -> None:
    model = build_model(
        {
            "name": "agb_net",
            "num_channels": 3,
            "num_classes": 5,
            "hidden_channels": 16,
            "num_blocks": 1,
            "conv_blocks": 2,
            "graph_order": 2,
            "dropout": 0.1,
        }
    )
    logits = model(torch.randn(2, 128, 3))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_ablation_flags_forward() -> None:
    variants = [
        {"use_graph": False},
        {"use_adaptive_graph": False},
        {"use_spatial_attention": False},
        {"use_temporal_attention": False},
        {"bidirectional": False},
        {
            "use_graph": False,
            "use_spatial_attention": False,
            "use_temporal_attention": False,
            "bidirectional": False,
        },
    ]
    for flags in variants:
        model = GraphBiGRUNet(
            num_channels=3,
            num_classes=5,
            hidden_channels=16,
            num_blocks=1,
            conv_blocks=2,
            **flags,
        )
        logits = model(torch.randn(2, 128, 3))
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()


def test_adaptive_adjacency_is_row_stochastic() -> None:
    model = GraphBiGRUNet(num_channels=6, num_classes=5, hidden_channels=16, num_blocks=1)
    adjacency = model._adjacency()
    assert adjacency.shape == (6, 6)
    row_sums = adjacency.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_adaptive_graph_is_learnable() -> None:
    model = GraphBiGRUNet(num_channels=4, num_classes=5, hidden_channels=16, num_blocks=1)
    assert any(name == "node_embeddings" for name, _ in model.named_parameters())
    fixed = GraphBiGRUNet(
        num_channels=4,
        num_classes=5,
        hidden_channels=16,
        num_blocks=1,
        use_adaptive_graph=False,
    )
    assert not any(name == "node_embeddings" for name, _ in fixed.named_parameters())


def test_rejects_invalid_graph_order() -> None:
    with pytest.raises(ValueError):
        GraphBiGRUNet(num_channels=3, num_classes=5, hidden_channels=16, graph_order=0)


def test_uses_group_norm_and_no_batch_norm() -> None:
    model = GraphBiGRUNet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=1)
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
