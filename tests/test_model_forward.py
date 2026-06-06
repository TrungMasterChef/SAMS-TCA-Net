import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import FCN1D, InceptionTimeBaseline, MambaSL, PatchTST, SimpleCNN1D, TSMixer, build_model


def test_baseline_forward_shapes() -> None:
    models = [
        SimpleCNN1D(num_channels=3, num_classes=5, hidden_channels=16),
        FCN1D(num_channels=3, num_classes=5, hidden_channels=16),
        InceptionTimeBaseline(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2),
        MambaSL(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2),
        PatchTST(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2, num_heads=2),
        TSMixer(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2),
    ]
    x = torch.randn(4, 256, 3)
    for model in models:
        logits = model(x)
        assert logits.shape == (4, 5)
        assert torch.isfinite(logits).all()


def test_baseline_forward_bct_shape() -> None:
    for model in (
        SimpleCNN1D(num_channels=3, num_classes=5, hidden_channels=16, input_layout="bct"),
        MambaSL(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2, input_layout="bct"),
        PatchTST(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2, num_heads=2, input_layout="bct"),
        TSMixer(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2, input_layout="bct"),
    ):
        logits = model(torch.randn(4, 3, 256))
        assert logits.shape == (4, 5)
        assert torch.isfinite(logits).all()


def test_model_factory_builds_every_model() -> None:
    names = [
        "msca_net",
        "simple_cnn_1d",
        "fcn_1d",
        "inception_time_baseline",
        "tcn_1d",
        "transformer",
        "mamba_sl",
        "patchtst",
        "tsmixer",
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
