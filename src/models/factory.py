"""Model factory for SAMS-TCA-Net and baseline comparisons."""

from __future__ import annotations

from torch import nn

from .baselines import FCN1D, InceptionTimeBaseline, ResNet1D, SimpleCNN1D
from .sams_tca_net import SAMSTCANet


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "sams_tca": SAMSTCANet,
    "sams_tca_net": SAMSTCANet,
    "samstcanet": SAMSTCANet,
    "simple_cnn_1d": SimpleCNN1D,
    "simplecnn1d": SimpleCNN1D,
    "fcn_1d": FCN1D,
    "fcn1d": FCN1D,
    "resnet_1d": ResNet1D,
    "resnet1d": ResNet1D,
    "inception_time_baseline": InceptionTimeBaseline,
    "inceptiontimebaseline": InceptionTimeBaseline,
}


def build_model(model_config: dict) -> nn.Module:
    """Instantiate a model from a config dictionary."""
    name = str(model_config.get("name", "sams_tca")).lower()
    model_cls = MODEL_REGISTRY.get(name)
    if model_cls is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model.name={name!r}. Available models: {available}")

    kwargs = {
        "num_channels": int(model_config["num_channels"]),
        "num_classes": int(model_config["num_classes"]),
        "hidden_channels": int(model_config.get("hidden_channels", 64)),
        "input_layout": str(model_config.get("input_layout", "btc")),
    }
    if model_cls in {SAMSTCANet, ResNet1D, InceptionTimeBaseline}:
        kwargs["num_blocks"] = int(model_config.get("num_blocks", 4))
    if model_cls in {SAMSTCANet, SimpleCNN1D, FCN1D, InceptionTimeBaseline}:
        kwargs["dropout"] = float(model_config.get("dropout", 0.1))
    if model_cls is SAMSTCANet:
        kwargs["use_sensor_attention"] = bool(model_config.get("use_sensor_attention", True))
        kwargs["use_scale_attention"] = bool(model_config.get("use_scale_attention", True))
        kwargs["use_temporal_channel_attention"] = bool(
            model_config.get("use_temporal_channel_attention", True)
        )
        kwargs["use_class_aware_pooling"] = bool(model_config.get("use_class_aware_pooling", True))
    return model_cls(**kwargs)
