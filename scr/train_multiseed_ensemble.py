from __future__ import annotations

from pathlib import Path
import os
import warnings

import joblib
import numpy as np
from sklearn.base import clone

from train_ensemble import (
    FEATURE_DIR,
    MODEL_DIR,
    build_models,
    cohort_weights,
    fit_estimator,
    infer_trial_ids,
    load_data,
    metric_line,
    predict_scores,
    split_iterator,
    trial_ranked_metric,
)


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

FEATURE_FILE = Path(os.environ.get("EEG_FEATURE_FILE", FEATURE_DIR / "X_rich_raw_subject.npy"))
ARTIFACT_PATH = Path(
    os.environ.get("EEG_MODEL_PATH", MODEL_DIR / "ensemble_rich_raw_subject_multiseed.pkl")
)
SEEDS = tuple(int(x) for x in os.environ.get("EEG_SEEDS", "7,19,42,73,101").split(","))
MODEL_NAMES = tuple(os.environ.get("EEG_MODEL_NAMES", "xgb,hist_gbdt").split(","))


def models_for_seed(seed: int) -> list[tuple[str, object]]:
    model_map = dict(build_models(seed=seed))
    return [(name, model_map[name]) for name in MODEL_NAMES]


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cohorts: np.ndarray | None,
    weights: np.ndarray | None,
) -> dict[str, object]:
    trial_ids = infer_trial_ids(groups)
    oof = np.zeros(len(y), dtype=np.float32)

    print(f"X: {X.shape}, y: {y.shape}, subjects: {len(np.unique(groups))}")
    print(f"models: {MODEL_NAMES}")
    print(f"seeds : {SEEDS}")

    for fold, (train_idx, val_idx) in enumerate(split_iterator(X, y, groups, cohorts, 5), start=1):
        print("\n" + "=" * 72)
        print(f"Fold {fold}")
        print("=" * 72)
        fold_scores: list[np.ndarray] = []
        for seed in SEEDS:
            for model_name, base_model in models_for_seed(seed):
                model = clone(base_model)
                train_weights = weights[train_idx] if weights is not None else None
                fit_estimator(model, X[train_idx], y[train_idx], train_weights)
                scores = predict_scores(model, X[val_idx])
                fold_scores.append(scores)

                acc, f1, window_acc, window_f1 = metric_line(y[val_idx], scores, groups[val_idx])
                trial_acc, trial_f1 = trial_ranked_metric(
                    y[val_idx],
                    scores,
                    groups[val_idx],
                    trial_ids[val_idx],
                )
                print(
                    f"{model_name}_{seed:<4d} threshold ACC/F1={acc:.4f}/{f1:.4f} "
                    f"window ranked ACC/F1={window_acc:.4f}/{window_f1:.4f} "
                    f"trial ranked ACC/F1={trial_acc:.4f}/{trial_f1:.4f}"
                )

        mean_scores = np.mean(np.stack(fold_scores, axis=0), axis=0)
        oof[val_idx] = mean_scores
        acc, f1, window_acc, window_f1 = metric_line(y[val_idx], mean_scores, groups[val_idx])
        trial_acc, trial_f1 = trial_ranked_metric(y[val_idx], mean_scores, groups[val_idx], trial_ids[val_idx])
        print(
            f"{'multiseed':12s} threshold ACC/F1={acc:.4f}/{f1:.4f} "
            f"window ranked ACC/F1={window_acc:.4f}/{window_f1:.4f} "
            f"trial ranked ACC/F1={trial_acc:.4f}/{trial_f1:.4f}"
        )

    acc, f1, window_acc, window_f1 = metric_line(y, oof, groups)
    trial_acc, trial_f1 = trial_ranked_metric(y, oof, groups, trial_ids)
    summary = {
        "threshold_acc": acc,
        "threshold_f1": f1,
        "window_rank_acc": window_acc,
        "window_rank_f1": window_f1,
        "trial_rank_acc": trial_acc,
        "trial_rank_f1": trial_f1,
        "model_names": list(MODEL_NAMES),
        "seeds": list(SEEDS),
    }

    print("\n" + "#" * 72)
    print("OOF Summary")
    print("#" * 72)
    print(
        f"multiseed threshold ACC/F1={acc:.4f}/{f1:.4f} "
        f"window ranked ACC/F1={window_acc:.4f}/{window_f1:.4f} "
        f"trial ranked ACC/F1={trial_acc:.4f}/{trial_f1:.4f}"
    )
    return summary


def train_final(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None, summary: dict[str, object]) -> None:
    fitted_models = []
    for seed in SEEDS:
        for model_name, base_model in models_for_seed(seed):
            model = clone(base_model)
            fit_estimator(model, X, y, weights)
            artifact_name = f"{model_name}_seed{seed}"
            fitted_models.append((artifact_name, model))
            print(f"trained: {artifact_name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature": FEATURE_FILE.stem,
            "models": fitted_models,
            "selected_model_names": [name for name, _ in fitted_models],
            "cv_summary": summary,
            "positive_per_test_subject": 4,
        },
        ARTIFACT_PATH,
    )
    print(f"Saved: {ARTIFACT_PATH}")


def main() -> None:
    # Keep train_ensemble.load_data's env-driven feature path behavior.
    X, y, groups, cohorts = load_data()
    weights = cohort_weights(cohorts)
    summary = run_cv(X, y, groups, cohorts, weights)
    train_final(X, y, weights, summary)


if __name__ == "__main__":
    main()
