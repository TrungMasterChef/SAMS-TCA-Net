# SAMS-TCA-Net

PyTorch implementation of SAMS-TCA-Net for accelerometer time-series classification.

## Structure

```text
src/models/sams_tca_net.py   # SAMSTCANet model
src/data/dataset.py          # X.npy/y.npy dataset, split, normalization, augmentation
src/train.py                 # training with CrossEntropyLoss
src/evaluate.py              # Accuracy, Macro-F1, Weighted-F1, confusion matrix
src/utils.py                 # config, metrics, dummy loaders
configs/sams_tca.yaml        # default config
tests/test_model_forward.py  # forward-pass tests
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The default config points to:

```text
src/data/Z24-dataset-processed/inputs.npy  # shape [N, C, T]
src/data/Z24-dataset-processed/labels.npy  # shape [N]
```

For your local copy of `thanglexuan/Z24-dataset-processed`, the detected shape is:

```text
inputs.npy: [1530, 27, 6000]
labels.npy: [1530]
```

Set `data.input_layout: nct` when the source array is `[N, C, T]`. Set it to `ntc` when the source array is `[N, T, C]`.

The dataset class creates deterministic train/val/test splits, computes mean/std from the train split only, converts source arrays to `[N, T, C]`, and returns tensors shaped `[T, C]`. PyTorch `DataLoader` batches are therefore `[B, T, C]`.

Augmentation options for training include jitter, scaling, and time masking.

The default configs use the crop-based preprocessing protocol:

```yaml
normalization: sample
window_mode: crop
window_length: 768
hop_length: 384
crop_mode: random_train_center_eval
temporal_stride: 1
taper: none
bandpass_filter: false
sampling_rate: 100.0
lowcut: 0.5
highcut: 40.0
filter_order: 4
transform: diff
eval_num_crops: 7
```

This trains on short random 768-sample windows, evaluates with seven deterministic crops averaged at logit level, normalizes each sample/window independently, and uses temporal differences rather than raw amplitude. The default configs use `model.num_channels: 27`.

The default SAMS-TCA-Net config also uses GroupNorm, label smoothing, gradient clipping, and ReduceLROnPlateau scheduling on validation Macro-F1 to reduce validation instability.

Splits are stratified by label at the original sequence level before crop/window extraction. Training augmentation is configurable through:

```yaml
augment: true
jitter_std: 0.01
scaling_std: 0.08
time_mask_ratio: 0.05
channel_mask_ratio: 0.15
```

This harder protocol reduces direct amplitude cues and randomly masks some sensor axes during training. It is intended to stress simple local CNN baselines and make sensor-axis and temporal-channel attention more useful.

## Train

```bash
python -m src.train --config configs/sams_tca.yaml
```

If `X.npy` and `y.npy` are not found, training falls back to dummy data so the pipeline can still run.

Training writes:

```text
outputs/sams_tca/history.csv
outputs/sams_tca/history.png
outputs/sams_tca/training_log.txt
outputs/sams_tca/best.pt
outputs/sams_tca/last.pt
outputs/sams_tca/val_confusion_matrix.npy
outputs/sams_tca/val_confusion_matrix.png
outputs/sams_tca/val_f1_scores.csv
```

The best checkpoint is selected by validation Macro-F1. Early stopping is controlled by `training.early_stopping_patience`.

Baseline configs are available under `configs/baselines/`:

```bash
python -m src.train --config configs/baselines/simple_cnn_1d.yaml
python -m src.train --config configs/baselines/fcn_1d.yaml
python -m src.train --config configs/baselines/resnet_1d.yaml
python -m src.train --config configs/baselines/inception_time_baseline.yaml
```

Each config selects the model through `model.name`.

Run SAMS-TCA-Net and all baseline configs sequentially:

```bash
python scripts/run_models.py
```

This writes a summary table to `outputs/model_results.csv`. Each model keeps its own artifacts, for example `outputs/baselines/simple_cnn_1d/`.

## Ablations

SAMS-TCA-Net ablation flags are configured under `model`:

```yaml
use_sensor_attention: true
use_scale_attention: true
use_temporal_channel_attention: true
use_class_aware_pooling: true
```

When `use_class_aware_pooling` is `false`, the model uses global average pooling plus a linear classifier.

Run all ablation configs and save metrics to `outputs/ablation_results.csv`:

```bash
python scripts/run_ablations.py
```

## Evaluate

```bash
python -m src.evaluate --config configs/sams_tca.yaml --checkpoint outputs/sams_tca/best.pt
```

The evaluation script reports Accuracy, Macro-F1, Weighted-F1, and the confusion matrix.

Evaluation writes:

```text
outputs/sams_tca/metrics.json
outputs/sams_tca/confusion_matrix.npy
outputs/sams_tca/confusion_matrix.png
outputs/sams_tca/f1_scores.csv
outputs/sams_tca/roc_curve.png
outputs/sams_tca/tsne.png
```

Metrics include the model parameter count and ROC-AUC scores (`roc_auc_macro`, `roc_auc_micro`, `roc_auc_weighted`).

## Test

```bash
pytest
```

## Model

`SAMSTCANet` accepts `[B, T, C]` by default and can also accept `[B, C, T]` with `input_layout="bct"`. The forward pass returns logits shaped `[B, num_classes]` and does not apply softmax.
