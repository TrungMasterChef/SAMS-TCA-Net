"""Model factory for MSCA-G and baseline comparisons."""

from __future__ import annotations

from torch import nn

from .baselines import (
    FCN1D,
    TCN1D,
    InceptionTimeBaseline,
    MambaSL,
    PatchTST,
    SimpleCNN1D,
    TSMixer,
    TransformerClassifier,
)
from .msca_net import MSCANet


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "msca_net": MSCANet,
    "mscanet": MSCANet,
    "msca": MSCANet,
    "msca_g": MSCANet,
    "simple_cnn_1d": SimpleCNN1D,
    "simplecnn1d": SimpleCNN1D,
    "fcn_1d": FCN1D,
    "fcn1d": FCN1D,
    "inception_time_baseline": InceptionTimeBaseline,
    "inceptiontimebaseline": InceptionTimeBaseline,
    "tcn_1d": TCN1D,
    "tcn": TCN1D,
    "transformer": TransformerClassifier,
    "transformer_classifier": TransformerClassifier,
    "mamba": MambaSL,
    "mamba_sl": MambaSL,
    "mambasl": MambaSL,
    "patchtst": PatchTST,
    "patch_tst": PatchTST,
    "tsmixer": TSMixer,
    "ts_mixer": TSMixer,
}


def build_model(model_config: dict) -> nn.Module:
    """Instantiate a model from a config dictionary."""
    name = str(model_config.get("name", "msca_net")).lower()
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
    if model_cls in {MSCANet, InceptionTimeBaseline, TCN1D, TransformerClassifier, MambaSL, PatchTST, TSMixer}:
        kwargs["num_blocks"] = int(model_config.get("num_blocks", 4))
    if model_cls in {MSCANet, SimpleCNN1D, FCN1D, InceptionTimeBaseline, TCN1D, TransformerClassifier, MambaSL, PatchTST, TSMixer}:
        kwargs["dropout"] = float(model_config.get("dropout", 0.1))
    if model_cls is MSCANet:
        kwargs["use_se"] = bool(model_config.get("use_se", True))
        kwargs["use_attention_pool"] = bool(model_config.get("use_attention_pool", True))
        kwargs["downsample"] = bool(model_config.get("downsample", True))
        kwargs["use_graph_front"] = bool(model_config.get("use_graph_front", False))
        kwargs["graph_embed_dim"] = int(model_config.get("graph_embed_dim", 10))
    if model_cls is MambaSL:
        kwargs["d_state"] = int(model_config.get("d_state", 16))
    if model_cls in {TransformerClassifier, PatchTST}:
        kwargs["num_heads"] = int(model_config.get("num_heads", 4))
        kwargs["mlp_ratio"] = float(model_config.get("mlp_ratio", 4.0))
    if model_cls is PatchTST:
        kwargs["patch_len"] = int(model_config.get("patch_len", 32))
        kwargs["patch_stride"] = int(model_config.get("patch_stride", 16))
    if model_cls is TSMixer:
        kwargs["num_tokens"] = int(model_config.get("num_tokens", 96))
    return model_cls(**kwargs)
