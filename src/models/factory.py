"""Model factory for SAMS-TCA-Net and baseline comparisons."""

from __future__ import annotations

from torch import nn

from .baselines import (
    FCN1D,
    LSTM1D,
    MLP1D,
    TCN1D,
    InceptionTimeBaseline,
    ResNet1D,
    SimpleCNN1D,
    TransformerClassifier,
)
from .graph_bigru import GraphBiGRUNet
from .msca_net import MSCANet
from .sams_tca_net import SAMSTCANet


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "sams_tca": SAMSTCANet,
    "sams_tca_net": SAMSTCANet,
    "samstcanet": SAMSTCANet,
    "agb_net": GraphBiGRUNet,
    "agbnet": GraphBiGRUNet,
    "graph_bigru": GraphBiGRUNet,
    "graphbigru": GraphBiGRUNet,
    "gcn_bigru": GraphBiGRUNet,
    "msca_net": MSCANet,
    "mscanet": MSCANet,
    "msca": MSCANet,
    "simple_cnn_1d": SimpleCNN1D,
    "simplecnn1d": SimpleCNN1D,
    "fcn_1d": FCN1D,
    "fcn1d": FCN1D,
    "resnet_1d": ResNet1D,
    "resnet1d": ResNet1D,
    "inception_time_baseline": InceptionTimeBaseline,
    "inceptiontimebaseline": InceptionTimeBaseline,
    "mlp_1d": MLP1D,
    "mlp": MLP1D,
    "lstm_1d": LSTM1D,
    "lstm": LSTM1D,
    "tcn_1d": TCN1D,
    "tcn": TCN1D,
    "transformer": TransformerClassifier,
    "transformer_classifier": TransformerClassifier,
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
    if model_cls in {SAMSTCANet, GraphBiGRUNet, MSCANet, ResNet1D, InceptionTimeBaseline, LSTM1D, TCN1D, TransformerClassifier}:
        kwargs["num_blocks"] = int(model_config.get("num_blocks", 4))
    if model_cls in {SAMSTCANet, GraphBiGRUNet, MSCANet, SimpleCNN1D, FCN1D, InceptionTimeBaseline, MLP1D, LSTM1D, TCN1D, TransformerClassifier}:
        kwargs["dropout"] = float(model_config.get("dropout", 0.1))
    if model_cls is SAMSTCANet:
        kwargs["use_sensor_attention"] = bool(model_config.get("use_sensor_attention", True))
        kwargs["use_scale_attention"] = bool(model_config.get("use_scale_attention", True))
        kwargs["use_temporal_channel_attention"] = bool(
            model_config.get("use_temporal_channel_attention", True)
        )
        kwargs["use_class_aware_pooling"] = bool(model_config.get("use_class_aware_pooling", True))
    if model_cls is GraphBiGRUNet:
        kwargs["conv_blocks"] = int(model_config.get("conv_blocks", 2))
        kwargs["graph_order"] = int(model_config.get("graph_order", 2))
        kwargs["node_embedding_dim"] = int(model_config.get("node_embedding_dim", 10))
        kwargs["use_graph"] = bool(model_config.get("use_graph", True))
        kwargs["use_adaptive_graph"] = bool(model_config.get("use_adaptive_graph", True))
        kwargs["use_spatial_attention"] = bool(model_config.get("use_spatial_attention", True))
        kwargs["use_temporal_attention"] = bool(model_config.get("use_temporal_attention", True))
        kwargs["bidirectional"] = bool(model_config.get("bidirectional", True))
    if model_cls is MSCANet:
        kwargs["use_se"] = bool(model_config.get("use_se", True))
        kwargs["use_attention_pool"] = bool(model_config.get("use_attention_pool", True))
        kwargs["downsample"] = bool(model_config.get("downsample", True))
    if model_cls is MLP1D:
        kwargs["pool_size"] = int(model_config.get("pool_size", 32))
    if model_cls in {LSTM1D}:
        kwargs["bidirectional"] = bool(model_config.get("bidirectional", True))
    if model_cls is TransformerClassifier:
        kwargs["num_heads"] = int(model_config.get("num_heads", 4))
        kwargs["mlp_ratio"] = float(model_config.get("mlp_ratio", 4.0))
    return model_cls(**kwargs)
