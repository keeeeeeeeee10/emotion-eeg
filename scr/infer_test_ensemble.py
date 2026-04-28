from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from features import extract_connectivity_features, extract_rich_features


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

TEST_DIR = Path("data/test")
MODEL_PATH = Path(os.environ.get("EEG_MODEL_PATH", "output/models/ensemble_rich_raw_subject.pkl"))
OUTPUT_XLSX = Path(os.environ.get("EEG_OUTPUT_XLSX", "output/submission_ensemble_rich_raw_subject.xlsx"))
OUTPUT_DEBUG_XLSX = Path(
    os.environ.get("EEG_OUTPUT_DEBUG_XLSX", "output/submission_ensemble_rich_raw_subject_debug.xlsx")
)
ENSEMBLE_METHOD = os.environ.get("EEG_ENSEMBLE_METHOD", "mean")

N_CHANNELS = 30
TRIAL_SAMPLES = 2500
N_TRIALS = 8
TOTAL_SAMPLES = TRIAL_SAMPLES * N_TRIALS
POSITIVE_PER_SUBJECT = 4


def _load_mat_test(mat_path: Path) -> dict[str, Any]:
    return loadmat(mat_path, squeeze_me=False, struct_as_record=False)


def _pick_eeg_array(data: dict[str, Any], mat_path: Path) -> np.ndarray:
    preferred_keys = ("test_eeg_c", "EEG_data", "eeg", "data")
    for key in preferred_keys:
        if key in data and isinstance(data[key], np.ndarray) and data[key].ndim == 2:
            return np.asarray(data[key])

    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return np.asarray(value)

    raise ValueError(f"{mat_path} has no 2D EEG array variable")


def _to_channel_first(arr: np.ndarray, mat_path: Path) -> np.ndarray:
    if arr.shape == (N_CHANNELS, TOTAL_SAMPLES):
        return arr
    if arr.shape == (TOTAL_SAMPLES, N_CHANNELS):
        return arr.T
    raise ValueError(
        f"{mat_path} bad EEG shape={arr.shape}, expected {(N_CHANNELS, TOTAL_SAMPLES)} "
        f"or {(TOTAL_SAMPLES, N_CHANNELS)}"
    )


def _zscore_per_trial_channel(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    return ((x - mean) / (std + 1e-6)).astype(np.float32, copy=False)


def _split_trials(eeg: np.ndarray) -> np.ndarray:
    trials = []
    for trial_idx in range(N_TRIALS):
        start = trial_idx * TRIAL_SAMPLES
        stop = (trial_idx + 1) * TRIAL_SAMPLES
        trial = eeg[:, start:stop]
        if trial.shape != (N_CHANNELS, TRIAL_SAMPLES):
            raise ValueError(f"Bad trial shape at trial={trial_idx + 1}: {trial.shape}")
        trials.append(trial.astype(np.float32, copy=False))
    return np.stack(trials, axis=0)


def _make_features(trials: np.ndarray, feature_name: str) -> np.ndarray:
    if feature_name in {
        "X_rich_raw",
        "X_rich_raw_subject",
        "X_rich_raw_subject_rank",
        "X_rich_conn_raw_subject",
        "X_rich_pseudo_subject",
        "X_rich_pseudo_subject_rank",
    }:
        X_raw = extract_rich_features(trials)
        if feature_name == "X_rich_conn_raw_subject":
            X_conn = extract_connectivity_features(trials)
            X_rich_conn = np.concatenate([X_raw, X_conn], axis=1).astype(np.float32, copy=False)
            X_norm = (X_rich_conn - X_rich_conn.mean(axis=0, keepdims=True)) / (
                X_rich_conn.std(axis=0, keepdims=True) + 1e-6
            )
            return np.concatenate([X_rich_conn, X_norm], axis=1).astype(np.float32, copy=False)
        if feature_name in {"X_rich_raw_subject", "X_rich_pseudo_subject"}:
            X_norm = (X_raw - X_raw.mean(axis=0, keepdims=True)) / (
                X_raw.std(axis=0, keepdims=True) + 1e-6
            )
            return np.concatenate([X_raw, X_norm], axis=1).astype(np.float32, copy=False)
        if feature_name in {"X_rich_raw_subject_rank", "X_rich_pseudo_subject_rank"}:
            X_norm = (X_raw - X_raw.mean(axis=0, keepdims=True)) / (
                X_raw.std(axis=0, keepdims=True) + 1e-6
            )
            order = np.argsort(X_raw, axis=0)
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order, np.arange(X_raw.shape[1])] = np.arange(len(X_raw), dtype=np.float32)[:, None]
            ranks = 2.0 * (ranks / float(len(X_raw) - 1)) - 1.0
            return np.concatenate([X_raw, X_norm, ranks], axis=1).astype(np.float32, copy=False)
        return X_raw

    trials_z = _zscore_per_trial_channel(trials)
    if feature_name == "X_rich_combo":
        X_z = extract_rich_features(trials_z)
        X_raw = extract_rich_features(trials)
        return np.concatenate([X_z, X_raw], axis=1).astype(np.float32, copy=False)

    return extract_rich_features(trials_z)


def _predict_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    scores = np.asarray(model.decision_function(X), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scores))


def _rank_average(score_matrix: np.ndarray) -> np.ndarray:
    ranked = np.zeros_like(score_matrix, dtype=np.float64)
    for row_idx, scores in enumerate(score_matrix):
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        if len(order) == 1:
            ranks[order] = 0.0
        else:
            ranks[order] = np.linspace(0.0, 1.0, len(order))
        ranked[row_idx] = ranks
    return ranked.mean(axis=0)


def _top_k_labels(scores: np.ndarray, k: int) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=np.int64)
    labels[np.argsort(scores)[-k:]] = 1
    return labels


def _natural_key(path: Path) -> int | str:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else path.stem


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Run scr/train_ensemble.py first.")

    test_files = sorted(TEST_DIR.glob("*.mat"), key=_natural_key)
    if not test_files:
        raise FileNotFoundError(f"No test .mat files found under {TEST_DIR}")

    artifact = joblib.load(MODEL_PATH)
    models = artifact["models"]
    feature_name = str(artifact.get("feature", "X_rich"))
    positive_per_subject = int(artifact.get("positive_per_test_subject", POSITIVE_PER_SUBJECT))

    rows: list[dict[str, int | str]] = []
    debug_rows: list[dict[str, float | int | str]] = []

    print(f"Found test files: {len(test_files)}")
    for mat_path in test_files:
        user_id = mat_path.stem
        mat = _load_mat_test(mat_path)
        eeg_raw = _pick_eeg_array(mat, mat_path)
        eeg = _to_channel_first(eeg_raw, mat_path)
        trials = _split_trials(eeg)
        X_feat = _make_features(trials, feature_name)

        model_scores = []
        for name, model in models:
            scores = _predict_scores(model, X_feat)
            model_scores.append(scores)
            print(f"{user_id} {name:12s}: {np.round(scores, 4)}")

        score_matrix = np.stack(model_scores, axis=0)
        if ENSEMBLE_METHOD == "rank":
            probs = _rank_average(score_matrix)
        else:
            probs = np.mean(score_matrix, axis=0)
        labels = _top_k_labels(probs, positive_per_subject)
        top_idx = np.flatnonzero(labels == 1)

        print(f"{user_id} ensemble    : {np.round(probs, 4)}")
        print(f"{user_id} top positive trials: {np.sort(top_idx + 1).tolist()}")

        for trial_id, (prob, label) in enumerate(zip(probs, labels), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "trial_id": int(trial_id),
                    "Emotion_label": int(label),
                }
            )
            row_debug: dict[str, float | int | str] = {
                "user_id": user_id,
                "trial_id": int(trial_id),
                "prob_positive": float(prob),
                "Emotion_label": int(label),
            }
            debug_rows.append(row_debug)

    df = pd.DataFrame(rows, columns=["user_id", "trial_id", "Emotion_label"])
    df_debug = pd.DataFrame(debug_rows)
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)
    df_debug.to_excel(OUTPUT_DEBUG_XLSX, index=False)

    print(f"Saved submission: {OUTPUT_XLSX}")
    print(f"Saved debug file: {OUTPUT_DEBUG_XLSX}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
