import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MSCANet, build_model


def test_forward_btc_shape() -> None:
    model = MSCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=3)
    logits = model(torch.randn(4, 256, 3))
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_forward_bct_shape() -> None:
    model = MSCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2, input_layout="bct")
    logits = model(torch.randn(4, 3, 256))
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_factory_builds_msca_net() -> None:
    model = build_model(
        {
            "name": "msca_net",
            "num_channels": 3,
            "num_classes": 5,
            "hidden_channels": 16,
            "num_blocks": 3,
            "dropout": 0.1,
        }
    )
    logits = model(torch.randn(2, 128, 3))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_ablation_flags_forward() -> None:
    variants = [
        {"use_se": False},
        {"use_attention_pool": False},
        {"downsample": False},
        {"use_se": False, "use_attention_pool": False, "downsample": False},
    ]
    for flags in variants:
        model = MSCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=3, **flags)
        logits = model(torch.randn(2, 128, 3))
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()


def test_uses_group_norm_and_no_batch_norm() -> None:
    model = MSCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=3)
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
