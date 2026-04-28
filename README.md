# EEG Emotion Recognition

Cross-subject EEG-based emotion recognition pipeline for the competition task.

The repository contains the training, feature extraction, ensembling, and inference code. Raw EEG data, generated features, model artifacts, submission spreadsheets, and contest PDF/XLSX files are intentionally excluded from git because they are large or local competition assets.

## Project Layout

```text
scr/
  build_dataset.py          Build window-level training arrays from data/train
  make_features.py          Build baseline rich features
  make_raw_features.py      Build current recommended raw/subject-normalized features
  train_ensemble.py         Train the recommended ensemble
  train_multiseed_ensemble.py
  train_pseudo_ensemble.py
  train_xgb.py              XGBoost baseline
  infer_test_ensemble.py    Generate the final test submission
RUN_ORDER.md                Detailed run order and experiment commands
```

Expected local data layout:

```text
data/
  train/
  test/
```

## Quick Start

```bash
conda activate eeg_emotion
python scr/build_dataset.py
python scr/make_features.py
python scr/make_raw_features.py
python scr/train_ensemble.py
python scr/infer_test_ensemble.py
```

The default inference output is:

```text
output/submission_ensemble_rich_raw_subject.xlsx
```

See `RUN_ORDER.md` for GPU options, optional experiments, and the current recommended submission path.

## GPU Notes

`train_ensemble.py`, `train_multiseed_ensemble.py`, `train_pseudo_ensemble.py`, and `train_xgb.py` use XGBoost CUDA automatically when a GPU is available.

Use these environment variables when needed:

```bash
EEG_USE_GPU=1 python scr/train_ensemble.py
EEG_USE_GPU=0 python scr/train_ensemble.py
EEG_MODEL_NAMES=all python scr/train_ensemble.py
```

LightGBM remains CPU-only unless the local environment is rebuilt with LightGBM GPU/CUDA support.
