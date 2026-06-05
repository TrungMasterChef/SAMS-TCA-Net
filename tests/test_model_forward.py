import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import FCN1D, InceptionTimeBaseline, ResNet1D, SAMSTCANet, SimpleCNN1D, build_model


def test_forward_btc_shape() -> None:
    model = SAMSTCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2)
    x = torch.randn(4, 128, 3)
    logits = model(x)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_forward_bct_shape() -> None:
    model = SAMSTCANet(
        num_channels=3,
        num_classes=5,
        hidden_channels=16,
        num_blocks=2,
        input_layout="bct",
    )
    x = torch.randn(4, 3, 128)
    logits = model(x)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_baseline_forward_shapes() -> None:
    models = [
        SimpleCNN1D(num_channels=3, num_classes=5, hidden_channels=16),
        FCN1D(num_channels=3, num_classes=5, hidden_channels=16),
        ResNet1D(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2),
        InceptionTimeBaseline(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2),
    ]
    x = torch.randn(4, 128, 3)
    for model in models:
        logits = model(x)
        assert logits.shape == (4, 5)
        assert torch.isfinite(logits).all()


def test_baseline_forward_bct_shape() -> None:
    model = SimpleCNN1D(num_channels=3, num_classes=5, hidden_channels=16, input_layout="bct")
    x = torch.randn(4, 3, 128)
    logits = model(x)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_model_factory_builds_every_model() -> None:
    names = [
        "sams_tca",
        "msca_net",
        "agb_net",
        "simple_cnn_1d",
        "fcn_1d",
        "resnet_1d",
        "inception_time_baseline",
        "mlp_1d",
        "lstm_1d",
        "tcn_1d",
        "transformer",
    ]
    for name in names:
        model = build_model(
            {
                "name": name,
                "num_channels": 3,
                "num_classes": 5,
                "hidden_channels": 16,
                "num_blocks": 2,
                "num_heads": 2,
                "dropout": 0.1,
            }
        )
        logits = model(torch.randn(2, 64, 3))
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()


def test_sams_tca_ablation_flags_forward() -> None:
    variants = [
        {"use_sensor_attention": False},
        {"use_scale_attention": False},
        {"use_temporal_channel_attention": False},
        {"use_class_aware_pooling": False},
        {
            "use_sensor_attention": False,
            "use_scale_attention": False,
            "use_temporal_channel_attention": False,
            "use_class_aware_pooling": False,
        },
    ]
    x = torch.randn(2, 64, 3)
    for flags in variants:
        model = SAMSTCANet(
            num_channels=3,
            num_classes=5,
            hidden_channels=16,
            num_blocks=2,
            **flags,
        )
        logits = model(x)
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()


def test_sams_tca_uses_group_norm_instead_of_batch_norm() -> None:
    model = SAMSTCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2)
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
