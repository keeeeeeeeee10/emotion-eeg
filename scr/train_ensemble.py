from __future__ import annotations

from itertools import combinations
from pathlib import Path
import os
import shutil
import warnings

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")
FEATURE_FILE = Path(os.environ.get("EEG_FEATURE_FILE", FEATURE_DIR / "X_rich_raw_subject.npy"))
ARTIFACT_PATH = Path(os.environ.get("EEG_MODEL_PATH", MODEL_DIR / "ensemble_rich_raw_subject.pkl"))
DEFAULT_MODEL_NAMES = ("xgb", "hist_gbdt")


def use_gpu() -> bool:
    mode = os.environ.get("EEG_USE_GPU", "auto").strip().lower()
    if mode in {"0", "false", "no", "cpu"}:
        return False
    if mode in {"1", "true", "yes", "gpu", "cuda"}:
        return True

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return shutil.which("nvidia-smi") is not None


def xgb_device_params() -> dict[str, str]:
    if use_gpu():
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist", "device": "cpu"}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    x_path = FEATURE_FILE
    y_path = FEATURE_DIR / "y.npy"
    g_path = FEATURE_DIR / "groups.npy"
    c_path = FEATURE_DIR / "cohorts.npy"

    if not x_path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {x_path}. Run scr/make_features.py and scr/make_raw_features.py first."
        )
    if not y_path.exists() or not g_path.exists():
        raise FileNotFoundError("Missing y/groups files. Run scr/build_dataset.py first.")

    X = np.load(x_path)
    y = np.load(y_path).astype(np.int64, copy=False)
    groups = np.load(g_path).astype(np.int64, copy=False)
    cohorts = np.load(c_path) if c_path.exists() else None
    return X, y, groups, cohorts


def cohort_weights(cohorts: np.ndarray | None) -> np.ndarray | None:
    if cohorts is None:
        return None
    cohorts = np.asarray(cohorts)
    values, counts = np.unique(cohorts, return_counts=True)
    weight_by_value = {value: len(cohorts) / (len(values) * count) for value, count in zip(values, counts)}
    return np.asarray([weight_by_value[v] for v in cohorts], dtype=np.float32)


def split_iterator(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cohorts: np.ndarray | None,
    n_splits: int,
):
    if cohorts is None:
        yield from GroupKFold(n_splits=n_splits).split(X, y, groups)
        return

    _, cohort_codes = np.unique(cohorts, return_inverse=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    yield from splitter.split(X, cohort_codes, groups)


def build_models(seed: int = 42) -> list[tuple[str, object]]:
    return [
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=700,
                learning_rate=0.025,
                num_leaves=15,
                max_depth=5,
                min_child_samples=18,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="binary",
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=550,
                max_depth=3,
                learning_rate=0.03,
                **xgb_device_params(),
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=3,
                reg_alpha=0.05,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gbdt",
            HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=420,
                max_leaf_nodes=15,
                l2_regularization=0.15,
                random_state=seed,
            ),
        ),
        (
            "svc_rbf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            C=1.7,
                            gamma="scale",
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.18,
                            penalty="l2",
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]


def filter_candidate_models(models: list[tuple[str, object]]) -> list[tuple[str, object]]:
    requested = os.environ.get("EEG_MODEL_NAMES", ",".join(DEFAULT_MODEL_NAMES)).strip()
    if not requested or requested.lower() == "all":
        return models

    names = tuple(name.strip() for name in requested.split(",") if name.strip())
    model_map = dict(models)
    missing = [name for name in names if name not in model_map]
    if missing:
        raise ValueError(f"Unknown model name(s) in EEG_MODEL_NAMES: {missing}. Available: {list(model_map)}")
    return [(name, model_map[name]) for name in names]


def fit_estimator(estimator: object, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> object:
    if weights is None:
        return estimator.fit(X, y)
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, model__sample_weight=weights)
    return estimator.fit(X, y, sample_weight=weights)


def predict_scores(estimator: object, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))[:, 1]
    scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scores))


def top_fraction_labels(scores: np.ndarray, groups: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=np.int64)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        k = int(round(len(idx) * fraction))
        if k <= 0:
            continue
        labels[idx[np.argsort(scores[idx])[-k:]]] = 1
    return labels


def infer_trial_ids(groups: np.ndarray, windows_per_trial: int = 5) -> np.ndarray:
    trial_ids = np.zeros(len(groups), dtype=np.int64)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) % windows_per_trial != 0:
            raise ValueError(
                f"Subject {group} has {len(idx)} windows, not divisible by {windows_per_trial}."
            )
        local_trials = np.arange(len(idx), dtype=np.int64) // windows_per_trial
        trial_ids[idx] = int(group) * 100 + local_trials
    return trial_ids


def trial_ranked_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    trial_ids: np.ndarray,
    positive_trials_per_subject: int = 4,
) -> tuple[float, float]:
    unique_trials = np.unique(trial_ids)
    trial_scores = np.zeros(len(unique_trials), dtype=np.float64)
    trial_labels = np.zeros(len(unique_trials), dtype=np.int64)
    trial_groups = np.zeros(len(unique_trials), dtype=np.int64)

    for out_idx, trial_id in enumerate(unique_trials):
        idx = np.flatnonzero(trial_ids == trial_id)
        trial_scores[out_idx] = float(np.mean(scores[idx]))
        trial_labels[out_idx] = int(round(float(np.mean(y_true[idx]))))
        trial_groups[out_idx] = int(groups[idx[0]])

    pred = np.zeros(len(unique_trials), dtype=np.int64)
    for group in np.unique(trial_groups):
        idx = np.flatnonzero(trial_groups == group)
        k = min(positive_trials_per_subject, len(idx))
        pred[idx[np.argsort(trial_scores[idx])[-k:]]] = 1

    return accuracy_score(trial_labels, pred), f1_score(trial_labels, pred)


def metric_line(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> tuple[float, float, float, float]:
    threshold_pred = (scores >= 0.5).astype(np.int64)
    ranked_pred = top_fraction_labels(scores, groups, fraction=0.5)
    return (
        accuracy_score(y_true, threshold_pred),
        f1_score(y_true, threshold_pred),
        accuracy_score(y_true, ranked_pred),
        f1_score(y_true, ranked_pred),
    )


def best_subset_by_oof(
    oof_scores: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    trial_ids: np.ndarray,
) -> tuple[list[str], dict[str, float]]:
    names = list(oof_scores)
    best_names: list[str] = names
    best_metrics = {
        "trial_rank_acc": -1.0,
        "trial_rank_f1": -1.0,
        "window_rank_acc": -1.0,
        "threshold_acc": -1.0,
    }

    for subset_size in range(1, len(names) + 1):
        for subset in combinations(names, subset_size):
            scores = np.mean(np.stack([oof_scores[name] for name in subset], axis=0), axis=0)
            threshold_acc, _, window_rank_acc, _ = metric_line(y, scores, groups)
            trial_rank_acc, trial_rank_f1 = trial_ranked_metric(y, scores, groups, trial_ids)
            candidate = {
                "trial_rank_acc": trial_rank_acc,
                "trial_rank_f1": trial_rank_f1,
                "window_rank_acc": window_rank_acc,
                "threshold_acc": threshold_acc,
            }
            best_key = (
                candidate["trial_rank_acc"],
                candidate["window_rank_acc"],
                candidate["threshold_acc"],
                -len(subset),
            )
            current_key = (
                best_metrics["trial_rank_acc"],
                best_metrics["window_rank_acc"],
                best_metrics["threshold_acc"],
                -len(best_names),
            )
            if best_key > current_key:
                best_names = list(subset)
                best_metrics = candidate

    return best_names, best_metrics


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    trial_ids: np.ndarray,
    cohorts: np.ndarray | None,
    weights: np.ndarray | None,
    n_splits: int = 5,
) -> dict[str, object]:
    models = filter_candidate_models(build_models())
    oof_scores = {name: np.zeros(len(y), dtype=np.float32) for name, _ in models}
    ensemble_scores = np.zeros(len(y), dtype=np.float32)

    print(f"X: {X.shape}, y: {y.shape}, subjects: {len(np.unique(groups))}")
    print(f"GPU mode: {'cuda' if use_gpu() else 'cpu'} for XGBoost")
    print(f"candidate models: {[name for name, _ in models]}")
    if cohorts is not None:
        values, counts = np.unique(cohorts, return_counts=True)
        print("cohorts:", dict(zip(values.tolist(), counts.tolist())))

    for fold, (train_idx, val_idx) in enumerate(split_iterator(X, y, groups, cohorts, n_splits), start=1):
        print("\n" + "=" * 72)
        print(f"Fold {fold}")
        print("=" * 72)
        print(f"train/val: {len(train_idx)}/{len(val_idx)} samples")
        print(f"val subjects: {len(np.unique(groups[val_idx]))}")

        fold_scores = []
        for name, base_estimator in models:
            estimator = clone(base_estimator)
            train_weights = weights[train_idx] if weights is not None else None
            fit_estimator(estimator, X[train_idx], y[train_idx], train_weights)
            scores = predict_scores(estimator, X[val_idx])
            oof_scores[name][val_idx] = scores
            fold_scores.append(scores)
            acc, f1, rank_acc, rank_f1 = metric_line(y[val_idx], scores, groups[val_idx])
            trial_acc, trial_f1 = trial_ranked_metric(
                y[val_idx],
                scores,
                groups[val_idx],
                trial_ids[val_idx],
            )
            print(
                f"{name:12s} threshold ACC/F1={acc:.4f}/{f1:.4f} "
                f"window ranked ACC/F1={rank_acc:.4f}/{rank_f1:.4f} "
                f"trial ranked ACC/F1={trial_acc:.4f}/{trial_f1:.4f}"
            )

        mean_scores = np.mean(np.stack(fold_scores, axis=0), axis=0)
        ensemble_scores[val_idx] = mean_scores
        acc, f1, rank_acc, rank_f1 = metric_line(y[val_idx], mean_scores, groups[val_idx])
        trial_acc, trial_f1 = trial_ranked_metric(
            y[val_idx],
            mean_scores,
            groups[val_idx],
            trial_ids[val_idx],
        )
        print(
            f"{'ensemble':12s} threshold ACC/F1={acc:.4f}/{f1:.4f} "
            f"window ranked ACC/F1={rank_acc:.4f}/{rank_f1:.4f} "
            f"trial ranked ACC/F1={trial_acc:.4f}/{trial_f1:.4f}"
        )

    print("\n" + "#" * 72)
    print("OOF Summary")
    print("#" * 72)
    summary: dict[str, object] = {}
    for name in [m[0] for m in models] + ["ensemble"]:
        scores = ensemble_scores if name == "ensemble" else oof_scores[name]
        acc, f1, rank_acc, rank_f1 = metric_line(y, scores, groups)
        trial_acc, trial_f1 = trial_ranked_metric(y, scores, groups, trial_ids)
        summary[f"{name}_rank_acc"] = rank_acc
        print(
            f"{name:12s} threshold ACC/F1={acc:.4f}/{f1:.4f} "
            f"window ranked ACC/F1={rank_acc:.4f}/{rank_f1:.4f} "
            f"trial ranked ACC/F1={trial_acc:.4f}/{trial_f1:.4f}"
        )

    best_names, best_metrics = best_subset_by_oof(oof_scores, y, groups, trial_ids)
    summary["selected_models"] = best_names
    summary["selected_metrics"] = best_metrics
    print("\nBest OOF subset by trial-ranked ACC")
    print(f"models: {best_names}")
    print(
        "trial ranked ACC/F1="
        f"{best_metrics['trial_rank_acc']:.4f}/{best_metrics['trial_rank_f1']:.4f}, "
        f"window ranked ACC={best_metrics['window_rank_acc']:.4f}, "
        f"threshold ACC={best_metrics['threshold_acc']:.4f}"
    )
    return summary


def train_final(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    summary: dict[str, object],
) -> None:
    print("\nTraining final rich-feature ensemble...")
    selected_names = set(summary.get("selected_models", [name for name, _ in build_models()]))
    fitted_models = []
    for name, base_estimator in build_models():
        if name not in selected_names:
            continue
        estimator = clone(base_estimator)
        fit_estimator(estimator, X, y, weights)
        fitted_models.append((name, estimator))
        print(f"trained: {name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "feature": FEATURE_FILE.stem,
        "models": fitted_models,
        "selected_model_names": [name for name, _ in fitted_models],
        "cv_summary": summary,
        "positive_per_test_subject": 4,
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved: {ARTIFACT_PATH}")


def main() -> None:
    X, y, groups, cohorts = load_data()
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(groups):
        raise ValueError(f"Bad shapes: X={X.shape}, y={y.shape}, groups={groups.shape}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or Inf")

    weights = cohort_weights(cohorts)
    trial_ids = infer_trial_ids(groups)
    summary = run_cv(X, y, groups, trial_ids, cohorts, weights, n_splits=5)
    train_final(X, y, weights, summary)


if __name__ == "__main__":
    main()
