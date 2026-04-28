# EEG Emotion Pipeline Run Order

This project should treat `测试结果模板.xlsx` only as an output-format example.
Do not use labels inside that file for training, validation, model selection, or submission correction.

## Recommended Full Rebuild

Run these commands from the project root:

```bash
conda activate eeg_emotion
python scr/build_dataset.py
python scr/make_features.py
python scr/make_raw_features.py
python scr/train_ensemble.py
python scr/infer_test_ensemble.py
```

`scr/train_ensemble.py`, `scr/train_multiseed_ensemble.py`, `scr/train_pseudo_ensemble.py`,
and the baseline `scr/train_xgb.py` use XGBoost on CUDA automatically when a GPU is available.
The default is:

```bash
EEG_USE_GPU=auto
```

You can force GPU or CPU explicitly:

```bash
EEG_USE_GPU=1 python scr/train_ensemble.py
EEG_USE_GPU=0 python scr/train_ensemble.py
```

Current environment note: XGBoost CUDA works. LightGBM GPU is not enabled in the installed
package, so LightGBM remains CPU-only unless the environment is rebuilt with LightGBM GPU/CUDA
support.

For speed, `scr/train_ensemble.py` trains the current recommended candidate set by default:

```text
xgb,hist_gbdt
```

`xgb` uses GPU when available; `hist_gbdt` is a CPU sklearn model. To rerun the slower full
model search over all candidates:

```bash
EEG_MODEL_NAMES=all python scr/train_ensemble.py
```

To train only GPU-backed XGBoost as a quick baseline:

```bash
EEG_MODEL_NAMES=xgb python scr/train_ensemble.py
```

The recommended submission file is:

```text
output/submission_ensemble_rich_raw_subject.xlsx
```

The debug file with probabilities is:

```text
output/submission_ensemble_rich_raw_subject_debug.xlsx
```

## Fast Re-Inference Only

Use this when features and model files already exist and you only want to regenerate test predictions:

```bash
conda activate eeg_emotion
python scr/infer_test_ensemble.py
```

## Optional Candidate Experiments

Pseudo-test feature experiment:

```bash
conda activate eeg_emotion
python scr/make_pseudo_features.py
python scr/train_pseudo_ensemble.py
EEG_MODEL_PATH=output/models/ensemble_rich_pseudo_subject_rank.pkl \
EEG_OUTPUT_XLSX=output/submission_ensemble_rich_pseudo_subject_rank.xlsx \
EEG_OUTPUT_DEBUG_XLSX=output/submission_ensemble_rich_pseudo_subject_rank_debug.xlsx \
python scr/infer_test_ensemble.py
```

Multi-seed stability experiment:

```bash
conda activate eeg_emotion
python scr/train_multiseed_ensemble.py
EEG_MODEL_PATH=output/models/ensemble_rich_raw_subject_multiseed.pkl \
EEG_OUTPUT_XLSX=output/submission_ensemble_rich_raw_subject_multiseed.xlsx \
EEG_OUTPUT_DEBUG_XLSX=output/submission_ensemble_rich_raw_subject_multiseed_debug.xlsx \
python scr/infer_test_ensemble.py
```

Connectivity-feature experiment:

```bash
conda activate eeg_emotion
EEG_FEATURE_FILE=output/features/X_rich_conn_raw_subject.npy \
EEG_MODEL_PATH=output/models/ensemble_rich_conn_raw_subject.pkl \
python scr/train_ensemble.py

EEG_MODEL_PATH=output/models/ensemble_rich_conn_raw_subject.pkl \
EEG_OUTPUT_XLSX=output/submission_ensemble_rich_conn_raw_subject.xlsx \
EEG_OUTPUT_DEBUG_XLSX=output/submission_ensemble_rich_conn_raw_subject_debug.xlsx \
python scr/infer_test_ensemble.py
```

## Current Recommendation

Keep `output/submission_ensemble_rich_raw_subject.xlsx` as the main submission unless an optional
experiment improves training-set Group CV and later platform feedback confirms the improvement.
