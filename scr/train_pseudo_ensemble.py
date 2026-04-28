from __future__ import annotations

from itertools import combinations
from pathlib import Path
import os
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from train_ensemble import build_models, cohort_weights, fit_estimator, predict_scores


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")
FEATURE_FILE = Path(os.environ.get("EEG_FEATURE_FILE", FEATURE_DIR / "X_rich_pseudo_subject_rank.npy"))
ARTIFACT_PATH = Path(
    os.environ.get("EEG_MODEL_PATH", MODEL_DIR / "ensemble_rich_pseudo_subject_rank.pkl")
)


def ranked_labels(scores: np.ndarray, rank_groups: np.ndarray, positives_per_group: int = 4) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=np.int64)
    for group in np.unique(rank_groups):
        idx = np.flatnonzero(rank_groups == group)
        k = min(positives_per_group, len(idx))
        pred[idx[np.argsort(scores[idx])[-k:]]] = 1
    return pred


def ranked_metrics(y: np.ndarray, scores: np.ndarray, rank_groups: np.ndarray) -> tuple[float, float]:
    pred = ranked_labels(scores, rank_groups)
    return accuracy_score(y, pred), f1_score(y, pred)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    X = np.load(FEATURE_FILE)
    y = np.load(FEATURE_DIR / "y_pseudo.npy").astype(np.int64, copy=False)
    cv_groups = np.load(FEATURE_DIR / "groups_pseudo_cv.npy").astype(np.int64, copy=False)
    rank_groups = np.load(FEATURE_DIR / "groups_pseudo_rank.npy").astype(np.int64, copy=False)
    cohort_path = FEATURE_DIR / "cohorts_pseudo.npy"
    cohorts = np.load(cohort_path) if cohort_path.exists() else None
    return X, y, cv_groups, rank_groups, cohorts


def best_subset(
    oof_scores: dict[str, np.ndarray],
    y: np.ndarray,
    rank_groups: np.ndarray,
) -> tuple[list[str], dict[str, float]]:
    names = list(oof_scores)
    best_names = names
    best_metrics = {"rank_acc": -1.0, "rank_f1": -1.0, "threshold_acc": -1.0}
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            scores = np.mean(np.stack([oof_scores[name] for name in subset], axis=0), axis=0)
            rank_acc, rank_f1 = ranked_metrics(y, scores, rank_groups)
            threshold_acc = accuracy_score(y, (scores >= 0.5).astype(np.int64))
            key = (rank_acc, threshold_acc, -len(subset))
            best_key = (best_metrics["rank_acc"], best_metrics["threshold_acc"], -len(best_names))
            if key > best_key:
                best_names = list(subset)
                best_metrics = {
                    "rank_acc": rank_acc,
                    "rank_f1": rank_f1,
                    "threshold_acc": threshold_acc,
                }
    return best_names, best_metrics


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_groups: np.ndarray,
    rank_groups: np.ndarray,
    cohorts: np.ndarray | None,
    weights: np.ndarray | None,
) -> dict[str, object]:
    models = build_models()
    oof_scores = {name: np.zeros(len(y), dtype=np.float32) for name, _ in models}
    splitter_target = y if cohorts is None else np.unique(cohorts, return_inverse=True)[1]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"X: {X.shape}, pseudo groups: {len(np.unique(rank_groups))}, subjects: {len(np.unique(cv_groups))}")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, splitter_target, cv_groups), start=1):
        print("\n" + "=" * 72)
        print(f"Fold {fold}")
        print("=" * 72)
        for name, base_model in models:
            model = clone(base_model)
            train_weights = weights[train_idx] if weights is not None else None
            fit_estimator(model, X[train_idx], y[train_idx], train_weights)
            scores = predict_scores(model, X[val_idx])
            oof_scores[name][val_idx] = scores
            rank_acc, rank_f1 = ranked_metrics(y[val_idx], scores, rank_groups[val_idx])
            threshold_acc = accuracy_score(y[val_idx], (scores >= 0.5).astype(np.int64))
            print(
                f"{name:12s} threshold ACC={threshold_acc:.4f} "
                f"pseudo-ranked ACC/F1={rank_acc:.4f}/{rank_f1:.4f}"
            )

        fold_scores = np.mean(np.stack([oof_scores[name][val_idx] for name, _ in models], axis=0), axis=0)
        rank_acc, rank_f1 = ranked_metrics(y[val_idx], fold_scores, rank_groups[val_idx])
        threshold_acc = accuracy_score(y[val_idx], (fold_scores >= 0.5).astype(np.int64))
        print(
            f"{'ensemble':12s} threshold ACC={threshold_acc:.4f} "
            f"pseudo-ranked ACC/F1={rank_acc:.4f}/{rank_f1:.4f}"
        )

    print("\n" + "#" * 72)
    print("OOF Summary")
    print("#" * 72)
    for name in [name for name, _ in models]:
        rank_acc, rank_f1 = ranked_metrics(y, oof_scores[name], rank_groups)
        threshold_acc = accuracy_score(y, (oof_scores[name] >= 0.5).astype(np.int64))
        print(
            f"{name:12s} threshold ACC={threshold_acc:.4f} "
            f"pseudo-ranked ACC/F1={rank_acc:.4f}/{rank_f1:.4f}"
        )

    selected, metrics = best_subset(oof_scores, y, rank_groups)
    print("\nBest OOF subset by pseudo-ranked ACC")
    print(f"models: {selected}")
    print(
        f"pseudo-ranked ACC/F1={metrics['rank_acc']:.4f}/{metrics['rank_f1']:.4f}, "
        f"threshold ACC={metrics['threshold_acc']:.4f}"
    )
    return {"selected_models": selected, "selected_metrics": metrics}


def train_final(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None, summary: dict[str, object]) -> None:
    selected = set(summary["selected_models"])
    fitted = []
    for name, base_model in build_models():
        if name not in selected:
            continue
        model = clone(base_model)
        fit_estimator(model, X, y, weights)
        fitted.append((name, model))
        print(f"trained: {name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature": FEATURE_FILE.stem,
            "models": fitted,
            "selected_model_names": [name for name, _ in fitted],
            "cv_summary": summary,
            "positive_per_test_subject": 4,
        },
        ARTIFACT_PATH,
    )
    print(f"Saved: {ARTIFACT_PATH}")


def main() -> None:
    X, y, cv_groups, rank_groups, cohorts = load_data()
    weights = cohort_weights(cohorts)
    summary = run_cv(X, y, cv_groups, rank_groups, cohorts, weights)
    train_final(X, y, weights, summary)


if __name__ == "__main__":
    main()
