# EEG Emotion Recognition for Competition Task 4

This repository contains a cleaned code release for the EEG positive/neutral emotion recognition pipeline used in Competition Task 4.

The final local validation result is:

```text
public-like subject-wise proxy accuracy = 0.8125
HC accuracy = 0.8563
DEP accuracy = 0.7250
```

Important: this is not the real public-test accuracy. The public test labels are unavailable. The score above is computed on a public-like proxy built from the labeled competition training set.

## What Is Included

```text
.
├── scripts/
│   ├── build_raw_trial_cache.py
│   ├── optimize_sklearn.py
│   ├── strict_seed_only.py
│   ├── run_seed_pairwise_ranking.py
│   ├── run_riemannian_baseline.py
│   ├── run_eegnet_pretrain_finetune.py
│   ├── run_eegnet_domain_adapt.py
│   ├── run_modma_softgate.py
│   ├── run_disease_aware_ensemble.py
│   ├── run_disease_pairwise_ranking.py
│   ├── run_subject_pairwise_reranker.py
│   ├── run_supcon_mlp_experiment.py
│   ├── run_ea_de_offset_experiment.py
│   ├── run_video_prototype_offset_experiment.py
│   └── run_weighted_offset_source_ensemble.py
├── seed_transfer/
│   ├── contest_data.py
│   ├── seed_data.py
│   ├── modma_data.py
│   ├── raw_trials.py
│   ├── features.py
│   ├── alignment.py
│   ├── riemannian.py
│   ├── torch_models.py
│   └── ...
├── docs/
│   └── research_method_results.html
├── requirements.txt
├── START_HERE.md
└── README.md
```

Generated outputs, datasets, checkpoints, Excel submissions, `.npz` prediction caches, and temporary experiment reports are intentionally not included.

## Expected Local Data Layout

Place or symlink the data folders under the repository root if you want to use default paths:

```text
.
├── SEED/
│   └── SEED_EEG/
├── MODMA/
└── 赛题四数据集及说明文档/
    ├── 训练集/
    ├── 公开测试集/
    ├── 数据集说明文档f.pdf
    └── 测试结果模板.xlsx
```

You can also pass explicit paths with script arguments such as `--seed-root`, `--contest-root`, `--modma-root`, and `--output-dir`.

## Install

Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
```

Core dependencies:

- `numpy`
- `scipy`
- `pandas`
- `openpyxl`
- `h5py`
- `scikit-learn`
- `joblib`
- `torch`

## Method Summary

The competition data contains both healthy controls and depressed subjects. The task is therefore not only positive/neutral emotion classification; it is also affected by cross-subject and HC/DEP heterogeneity.

The final strategy is:

1. Convert all raw EEG into 30-channel, 10-second trials.
2. Extract bandpower/differential-entropy style summary features.
3. Train several complementary branches:
   - EEGNet / EEGNet-CORAL.
   - Riemannian baseline.
   - SEED-informed pairwise ranking.
   - MODMA-informed disease gate.
   - Disease-aware source ensemble.
   - SupCon MLP.
   - Video prototype branch.
4. Use subject-wise Top-K ranking rather than a global threshold.
5. Build a public-like proxy from training videos:
   - each 50-second training video is split into five 10-second offsets;
   - offsets 1/3/4 with weights 0.4/0.4/0.2 are used to emulate the 10-second public-test format.
6. Select final source weights inside subject-wise outer folds.

Final fixed source weights:

```text
0.30 * predictions_disease_aware_ensemble.npz
0.30 * predictions_supcon_mlp_main_h96.npz
0.20 * predictions_eegnet_coral.npz
0.20 * predictions_video_prototype_offset_main_proto.npz
```

## Rebuild Order

Run commands from the repository root.

### 1. Build raw trial caches

```powershell
python scripts\build_raw_trial_cache.py
```

This creates raw 30 x 2500 trial caches under `outputs/`.

### 2. Build summary features and classical baseline

```powershell
python scripts\optimize_sklearn.py
```

This creates the contest and public trial-summary feature caches used by later scripts.

### 3. Build SEED strict summary cache and SEED pairwise branch

```powershell
python scripts\strict_seed_only.py
python scripts\run_seed_pairwise_ranking.py
```

### 4. Train Riemannian branch

```powershell
python scripts\run_riemannian_baseline.py
```

### 5. Train EEGNet branches

```powershell
python scripts\run_eegnet_pretrain_finetune.py

python scripts\run_eegnet_domain_adapt.py --domain-loss coral

python scripts\run_eegnet_domain_adapt.py --domain-loss mmd
```

The final model uses the CORAL branch through `predictions_eegnet_coral.npz`.

### 6. Train MODMA disease gate and disease-aware ensemble

```powershell
python scripts\run_modma_softgate.py

python scripts\run_disease_pairwise_ranking.py

python scripts\run_disease_aware_ensemble.py
```

### 7. Train subject pairwise reranker

```powershell
python scripts\run_subject_pairwise_reranker.py
```

### 8. Train SupCon MLP branch

```powershell
python scripts\run_supcon_mlp_experiment.py `
  --tag main_h96 `
  --outer-seeds 2026,2031,2042 `
  --epochs 100 `
  --patience 18 `
  --hidden-dim 96 `
  --embed-dim 24 `
  --supcon-weight 0.15 `
  --temperature 0.15 `
  --dropout 0.35 `
  --lr 0.001 `
  --weight-decay 0.002 `
  --torch-threads 4
```

### 9. Build EA-DE feature cache

```powershell
python scripts\run_ea_de_offset_experiment.py `
  --tag main_ea_de `
  --outer-seeds 2026 `
  --folds 5 `
  --inner-folds 3 `
  --offsets 1,3 `
  --public-offset-weights 0.5,0.5 `
  --feature-sets summary,ea_multi `
  --norm-modes subject_center `
  --reducers select160,pca64 `
  --models logreg_c0.03,extra_trees `
  --dep-weights 1.0,2.0 `
  --window-seconds 1,2,4
```

This creates:

```text
outputs/cache_ea_de_multifeature_ws1p0_2p0_4p0.npz
```

### 10. Train video prototype branch

```powershell
python scripts\run_video_prototype_offset_experiment.py `
  --tag main_proto `
  --outer-seeds 2026,2031,2042 `
  --folds 5 `
  --offsets 1,3,4 `
  --public-offset-weights 0.4,0.4,0.2 `
  --feature-sets summary,multi,ea_multi `
  --norm-modes subject_center,subject_zscore `
  --reducers select80,select160,select320,pca64 `
  --metrics cosine,euclidean `
  --aggregations mean,max,logsum
```

This creates:

```text
outputs/predictions_video_prototype_offset_main_proto.npz
```

### 11. Final weighted-offset source ensemble

```powershell
python scripts\run_weighted_offset_source_ensemble.py `
  --tag main_w0134 `
  --outer-seeds 2026,2031,2042 `
  --folds 5 `
  --offsets 1,3,4 `
  --offset-weights 0.4,0.4,0.2 `
  --combo-max-size 4 `
  --weight-step 0.1 `
  --sources predictions_disease_aware_ensemble.npz,predictions_supcon_mlp_main_h96.npz,predictions_eegnet_coral.npz,predictions_modma_softgate.npz,predictions_subject_pairwise_reranker.npz,predictions_video_prototype_offset_main_proto.npz
```

Final submission file:

```text
outputs/weighted_offset_source_ensemble_main_w0134/public_test_submission_weighted_offset_source_ensemble_main_w0134_top4.xlsx
```

Final report:

```text
outputs/weighted_offset_source_ensemble_main_w0134/weighted_offset_source_ensemble_main_w0134_report.xlsx
outputs/weighted_offset_source_ensemble_main_w0134/weighted_offset_source_ensemble_main_w0134_summary.json
```

## Validation Result

The final local validation result from `weighted_offset_source_ensemble_main_w0134`:

| Metric | Value |
|---|---:|
| seed 2026 nested accuracy | 0.8000 |
| seed 2031 nested accuracy | 0.8250 |
| seed 2042 nested accuracy | 0.8167 |
| mean-rank nested accuracy | 0.8125 |
| HC accuracy | 0.8563 |
| DEP accuracy | 0.7250 |
| full-proxy fixed config accuracy | 0.8333 |

The public test submission is generated with exactly 4 positive labels per `P_test` subject.

## Important Notes

- Do not treat the 50-second full-video aggregation score as the expected public-test score.
- The public test has only 10 seconds per video, so the public-like offset proxy is more realistic.
- Do not split trials from the same subject across train/validation folds.
- Do not train on public-test feedback.
- The final `0.8125` is a local proxy result, not an official leaderboard result.

## Documentation

Open this file for a Chinese visual explanation of the research idea, method diagram, validation protocol, and final result:

```text
docs/research_method_results.html
```
