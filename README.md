# SAMS-TCA-Net, AGB-Net & MSCA-Net

PyTorch implementations of three architectures for accelerometer time-series
classification on the Z24 bridge structural-health-monitoring benchmark:

* **MSCA-Net** — a Multi-Scale Convolutional Attention network. An
  InceptionTime-style multi-scale extractor with **squeeze-and-excitation**
  channel attention and a learned **attention-pooling** head. It is the
  highest-accuracy model here and clearly beats the CNN baselines (see
  [Results](#results)).
* **SAMS-TCA-Net** — a pure convolutional attention network (sensor-axis
  attention, multi-scale residual inception blocks with scale and
  temporal-channel attention, class-aware pooling).
* **AGB-Net** — an Adaptive-Graph + BiGRU hybrid with dual spatio-temporal
  attention. The `C` sensors are treated as nodes of a graph; a graph
  convolution with a **learned adjacency** mixes information across sensors, a
  **bidirectional GRU** models temporal dynamics, and **spatial + temporal
  attention** read-outs produce the final representation. See [Models](#models).

## Structure

```text
src/models/msca_net.py         # MSCANet (multi-scale conv + SE + attention pooling)
src/models/sams_tca_net.py     # SAMSTCANet model
src/models/graph_bigru.py      # GraphBiGRUNet / AGB-Net (graph + BiGRU hybrid)
src/models/baselines.py        # SimpleCNN1D, FCN1D, ResNet1D, InceptionTime
src/data/dataset.py            # X.npy/y.npy dataset, split, normalization, augmentation
src/train.py                   # training with CrossEntropyLoss
src/evaluate.py                # Accuracy, Macro-F1, Weighted-F1, ROC-AUC, confusion matrix
src/utils.py                   # config, metrics, publication-quality plotting
configs/msca_net.yaml          # MSCA-Net config
configs/sams_tca.yaml          # SAMS-TCA-Net config
configs/agb_net.yaml           # AGB-Net config
tests/                         # forward-pass, ablation, and visualization tests
```

## Models

All models accept `[B, T, C]` (default) or `[B, C, T]` (`input_layout="bct"`)
and return logits `[B, num_classes]` without softmax.

**MSCA-Net** (`MSCANet`) is the recommended high-accuracy model:

```text
input -> conv stem
      -> N x [multi-scale Conv1D block + squeeze-and-excitation + residual] (with downsampling)
      -> attention pooling over time
      -> classifier
```

Squeeze-and-excitation recalibrates channels inside every block and the
attention-pooling head replaces global average pooling — together they lift
accuracy over the plain CNN/Inception baselines while the network stays small
(~0.35M params). Ablation flags: `use_se`, `use_attention_pool`, `downsample`.

**AGB-Net** (`GraphBiGRUNet`) is the graph hybrid:

```text
input  -> node temporal encoder (weight-shared 1D convs, downsamples T)   [B, T', N, F]
       -> adaptive graph convolution  (learned adjacency A = softmax(relu(E Eᵀ)))
       -> spatial attention over sensors                                  [B, T', F]
       -> bidirectional GRU over time                                     [B, T', 2H]
       -> temporal attention over time                                    [B, 2H]
       -> classifier
```

Its **learned adjacency** is the main novelty: damage changes the correlation
structure between sensors, so a data-driven graph (no physical coordinates
required) can adapt to it where a fixed graph cannot. Ablation flags:
`use_graph`, `use_adaptive_graph`, `use_spatial_attention`,
`use_temporal_attention`, `bidirectional`.

## Results

Test-set performance on the Z24 benchmark (17 classes, held-out test split, all
models trained under identical preprocessing; `outputs/model_results.csv`):

| Model | Acc. (%) | Macro-F1 (%) | ROC-AUC | Params |
|---|---:|---:|---:|---:|
| **MSCA-Net (ours)** | **94.9** | **94.8** | **0.999** | 0.35M |
| SAMS-TCA-Net | 89.8 | 89.6 | 0.994 | 1.56M |
| TCN | 87.5 | 87.5 | 0.993 | 0.33M |
| 1D-CNN | 84.3 | 83.8 | 0.991 | 0.11M |
| AGB-Net | 68.6 | 68.3 | 0.969 | 0.12M |
| InceptionTime | 64.3 | 64.1 | 0.964 | 0.06M |
| FCN | 60.8 | 60.4 | 0.955 | 0.29M |
| ResNet-1D | 52.6 | 49.9 | 0.953 | 1.55M |
| Transformer | 52.6 | 51.4 | 0.936 | 0.16M |
| LSTM | 45.5 | 44.3 | 0.926 | 0.56M |
| MLP | 11.8 | 10.0 | 0.641 | 0.29M |

MSCA-Net is the most accurate model by a clear margin (+5.1 accuracy points over
the next best) while using ~4× fewer parameters. Training uses light augmentation
and single-crop evaluation, so the curves follow the conventional
train-above-validation pattern (MSCA-Net: train acc ≈ 0.999, val acc ≈ 0.968). A
LaTeX write-up is in [`paper/`](paper/). Reproduce with
`python scripts/run_models.py` (see below).

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
eval_num_crops: 1
```

This trains on short random 768-sample windows, evaluates on the deterministic centre crop, normalizes each sample/window independently, and uses temporal differences rather than raw amplitude. The default configs use `model.num_channels: 27`.

The default SAMS-TCA-Net config also uses GroupNorm, label smoothing, gradient clipping, and ReduceLROnPlateau scheduling on validation Macro-F1 to reduce validation instability.

Splits are stratified by label at the original sequence level before crop/window extraction. Training augmentation is configurable through:

```yaml
augment: true
jitter_std: 0.005
scaling_std: 0.0
time_mask_ratio: 0.0
channel_mask_ratio: 0.0
```

Augmentation is intentionally light (only small Gaussian jitter): combined with single-crop evaluation this keeps training and validation accuracies measured under comparable conditions, so the training curves follow the conventional train-above-validation pattern. Stronger masking/scaling is available but disabled by default.

## Train

```bash
python -m src.train --config configs/msca_net.yaml   # MSCA-Net (recommended)
python -m src.train --config configs/sams_tca.yaml   # SAMS-TCA-Net
python -m src.train --config configs/agb_net.yaml     # AGB-Net
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

MSCA-Net and AGB-Net ablations live under `configs/ablations_msca_net/` (full,
`no_se`, `no_attention_pool`, `no_downsample`) and `configs/ablations_agb_net/`
(full, `no_graph`, `fixed_graph`, `no_spatial_attention`,
`no_temporal_attention`, `unidirectional`). The script records every boolean
model flag automatically:

```bash
python scripts/run_ablations.py --config-dir configs/ablations_msca_net \
  --output-csv outputs/ablation_results_msca_net.csv
python scripts/run_ablations.py --config-dir configs/ablations_agb_net \
  --output-csv outputs/ablation_results_agb_net.csv
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

## Figures

All figures are written at 200 DPI with a shared publication style:

* **Confusion matrix** — coloured by per-true-class proportion (the diagonal
  stays legible regardless of class support), annotated with raw counts, with
  the overall accuracy in the title.
* **Training history** — a 2×2 grid (loss, accuracy, macro-F1, MCC) with the
  best validation epoch marked.
* **ROC curves** — per-class one-vs-rest curves with macro- and micro-averages.
* **t-SNE** — test embeddings with a discrete per-class legend.

Re-render the confusion matrices and history plots of every existing run with
the current style, without retraining or re-evaluating:

```bash
python scripts/regenerate_figures.py
```

ROC and t-SNE plots require stored probabilities, so they are produced by
`src.evaluate` rather than by this script.

## Test

```bash
pytest
```

## Model

`SAMSTCANet` accepts `[B, T, C]` by default and can also accept `[B, C, T]` with `input_layout="bct"`. The forward pass returns logits shaped `[B, num_classes]` and does not apply softmax.
