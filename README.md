# MSCA-G

PyTorch implementation of **MSCA-G** — a Multi-Scale Convolutional Attention
network with an **adaptive sensor graph** — for accelerometer time-series
classification on the Z24 bridge structural-health-monitoring benchmark, together
with eight standard time-series baselines for a fair comparison.

MSCA-G mixes information across the 27 accelerometers through a *learned*
cross-sensor graph, extracts features with an InceptionTime-style multi-scale
backbone and **squeeze-and-excitation** channel attention, and aggregates over
time with a learned **attention-pooling** head. It reaches the highest accuracy
here while staying small (~0.35M parameters). See [Results](#results).

## Structure

```text
src/models/msca_net.py    # MSCANet / MSCA-G (graph + multi-scale conv + SE + attention pooling)
src/models/baselines.py   # MLP, SimpleCNN, FCN, ResNet, InceptionTime, LSTM, TCN, Transformer
src/models/factory.py     # build_model from a config dict
src/data/dataset.py       # X.npy/y.npy dataset, split, normalization, augmentation
src/train.py              # training with CrossEntropyLoss
src/evaluate.py           # Accuracy, Macro-F1, Weighted-F1, ROC-AUC, confusion matrix
src/utils.py              # config, metrics, publication-quality plotting
configs/msca_net.yaml     # MSCA-G config
configs/baselines/        # one config per baseline
configs/ablations_msca_net/  # MSCA-G ablation configs
tests/                    # forward-pass, ablation, and visualization tests
```

## Model

All models accept `[B, T, C]` (default) or `[B, C, T]` (`input_layout="bct"`)
and return logits `[B, num_classes]` without softmax. **MSCA-G** is
`MSCANet` with `use_graph_front: true`:

```text
input -> adaptive sensor graph (learned adjacency A = softmax(relu(E Eᵀ)), residual mix)
      -> conv stem
      -> N x [multi-scale Conv1D block + squeeze-and-excitation + residual] (with downsampling)
      -> attention pooling over time
      -> classifier
```

The **adaptive sensor graph** learns the cross-sensor connectivity from data
(damage changes inter-sensor correlations, so a fixed wiring cannot capture it),
squeeze-and-excitation recalibrates channels inside every block, and the
attention-pooling head replaces global average pooling. Ablation flags:
`use_graph_front`, `use_se`, `use_attention_pool`, `downsample`.

## Results

Test-set performance on the Z24 benchmark (17 classes, held-out test split, all
models trained under the **same** preprocessing and training schedule;
`outputs/model_results.csv`):

| Model | Acc. (%) | Macro-F1 (%) | ROC-AUC | Params |
|---|---:|---:|---:|---:|
| **MSCA-G (ours)** | **97.2** | **97.2** | **1.000** | 0.35M |
| ResNet-1D | 96.9 | 96.8 | 0.999 | 1.55M |
| InceptionTime | 93.3 | 93.3 | 0.999 | 0.06M |
| FCN | 92.2 | 92.2 | 0.994 | 0.29M |
| TCN | 87.5 | 87.7 | 0.992 | 0.33M |
| 1D-CNN | 78.4 | 78.2 | 0.984 | 0.11M |
| Transformer | 69.0 | 69.0 | 0.964 | 0.16M |
| LSTM | 39.6 | 39.2 | 0.904 | 0.56M |
| MLP | 14.1 | 12.3 | 0.650 | 0.29M |

All models use the **same** preprocessing and training schedule (epoch=100). The
closest competitor is a well-tuned ResNet-1D (96.9%), which MSCA-G matches and
slightly exceeds using ~4× fewer parameters; the adaptive sensor graph alone adds
+3.5 points (ablation) for ~1k extra parameters. Training uses light augmentation
and single-crop evaluation, so the curves follow the conventional
train-above-validation pattern (MSCA-G: train acc ≈ 0.997, val acc ≈ 0.982). A
LaTeX write-up is in [`paper/`](paper/).

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

The detected shape for `thanglexuan/Z24-dataset-processed` is `inputs.npy:
[1530, 27, 6000]`, `labels.npy: [1530]`. Set `data.input_layout: nct` when the
source array is `[N, C, T]` (or `ntc` for `[N, T, C]`). Splits are deterministic
and stratified by label at the sequence level; train statistics are computed from
the train split only.

All configs share one **fair-comparison** preprocessing and training schedule:

```yaml
# data
normalization: sample      # per-window standardisation
transform: diff            # first-order temporal difference
window_mode: crop
window_length: 768         # random crop (train) / centre crop (eval)
crop_mode: random_train_center_eval
eval_num_crops: 1
augment: true              # light only:
jitter_std: 0.005
scaling_std: 0.0
time_mask_ratio: 0.0
channel_mask_ratio: 0.0
# training
batch_size: 64
epochs: 100
learning_rate: 0.0008
weight_decay: 0.0001
early_stopping_patience: 20
scheduler: reduce_on_plateau
label_smoothing: 0.05
gradient_clip_norm: 1.0
```

Augmentation is intentionally light and evaluation is single-crop, so training
and validation accuracies are measured under comparable conditions (the curves
follow the conventional train-above-validation pattern). If `inputs.npy`/
`labels.npy` are absent, training falls back to dummy data so the pipeline still
runs.

## Train

```bash
python -m src.train --config configs/msca_net.yaml                    # MSCA-G (recommended)
python -m src.train --config configs/baselines/tcn_1d.yaml           # any baseline
```

The best checkpoint is selected by validation Macro-F1 with early stopping.
Training writes `history.csv`, `history.png`, `training_log.txt`, `best.pt`,
`last.pt`, and validation confusion-matrix artifacts under the config's output
directory (e.g. `outputs/msca_net/`).

Run MSCA-G and all baselines sequentially and write a summary table to
`outputs/model_results.csv`:

```bash
python scripts/run_models.py
```

## Ablations

MSCA-G ablation configs live under `configs/ablations_msca_net/` (`full`,
`no_graph`, `no_se`, `no_attention_pool`, `no_downsample`). The script records
every boolean model flag automatically:

```bash
python scripts/run_ablations.py --config-dir configs/ablations_msca_net \
  --output-csv outputs/ablation_results_msca_net.csv
```

## Evaluate

```bash
python -m src.evaluate --config configs/msca_net.yaml --checkpoint outputs/msca_net/best.pt
```

Reports Accuracy, Macro-F1, Weighted-F1, ROC-AUC, and the confusion matrix, and
writes `metrics.json`, `confusion_matrix.{npy,png}`, `f1_scores.csv`,
`roc_curve.png`, and `tsne.png` under the config's output directory.

## Figures

All figures are written at 300 DPI with a shared, publication-grade style:

* **Confusion matrix** — coloured by per-true-class proportion (the diagonal
  stays legible regardless of class support), annotated with raw counts, overall
  accuracy in the title.
* **Training history** — a 2×2 grid (loss, accuracy, macro-F1, MCC) with the best
  validation epoch marked.
* **ROC curves** — per-class one-vs-rest curves with macro- and micro-averages.
* **t-SNE** — learned features with a discrete per-class legend.

Re-render saved confusion matrices and history plots with the current style
(without retraining):

```bash
python scripts/regenerate_figures.py
```

ROC and t-SNE plots require stored probabilities, so they are produced by
`src.evaluate`.

## Test

```bash
pytest
```
