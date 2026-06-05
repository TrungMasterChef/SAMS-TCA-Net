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

The dataset directory should contain:

```text
X.npy  # shape [N, T, C]
y.npy  # shape [N]
```

For your Hugging Face dataset `thanglexuan/Z24-dataset-processed`, place or download the processed files into the path configured by `data.data_dir`, for example:

```text
data/Z24-dataset-processed/X.npy
data/Z24-dataset-processed/y.npy
```

The dataset class creates deterministic train/val/test splits, computes mean/std from the train split only, and returns tensors shaped `[T, C]`. PyTorch `DataLoader` batches are therefore `[B, T, C]`.

Augmentation options for training include jitter, scaling, and time masking.

## Train

```bash
python -m src.train --config configs/sams_tca.yaml
```

If `X.npy` and `y.npy` are not found, training falls back to dummy data so the pipeline can still run.

## Evaluate

```bash
python -m src.evaluate --config configs/sams_tca.yaml --checkpoint outputs/sams_tca_net.pt
```

The evaluation script reports Accuracy, Macro-F1, Weighted-F1, and the confusion matrix.

## Test

```bash
pytest
```

## Model

`SAMSTCANet` accepts `[B, T, C]` by default and can also accept `[B, C, T]` with `input_layout="bct"`. The forward pass returns logits shaped `[B, num_classes]` and does not apply softmax.
