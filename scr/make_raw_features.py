from __future__ import annotations

from pathlib import Path

import numpy as np

from build_dataset import (
    KEY_TO_LABEL,
    N_CHANNELS,
    N_TRIALS_PER_EMOTION,
    N_WINDOWS_PER_TRIAL,
    TRIAL_SAMPLES,
    WINDOW_SAMPLES,
    _ensure_channel_first,
    _load_mat_auto,
    iter_subjects,
)
from features import check_feature_matrix, extract_connectivity_features, extract_rich_features


TRAIN_ROOT = Path("data/train")
FEATURE_DIR = Path("output/features")
BATCH_SIZE = 96


def subject_normalize(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X_norm = np.zeros_like(X, dtype=np.float32)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        mean = X[idx].mean(axis=0, keepdims=True)
        std = X[idx].std(axis=0, keepdims=True)
        X_norm[idx] = (X[idx] - mean) / (std + 1e-6)
    return X_norm


def subject_rank_normalize(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X_rank = np.zeros_like(X, dtype=np.float32)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        order = np.argsort(X[idx], axis=0)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order, np.arange(X.shape[1])] = np.arange(len(idx), dtype=np.float32)[:, None]
        if len(idx) > 1:
            ranks = ranks / float(len(idx) - 1)
        X_rank[idx] = 2.0 * ranks - 1.0
    return X_rank


def flush_batch(batch: list[np.ndarray], rich_out: list[np.ndarray], conn_out: list[np.ndarray]) -> None:
    if not batch:
        return
    X_batch = np.stack(batch, axis=0).astype(np.float32, copy=False)
    rich_out.append(extract_rich_features(X_batch))
    conn_out.append(extract_connectivity_features(X_batch))
    batch.clear()


def main() -> None:
    subjects = iter_subjects(TRAIN_ROOT)
    if not subjects:
        raise FileNotFoundError(f"No mat files found under {TRAIN_ROOT}")

    feature_chunks: list[np.ndarray] = []
    conn_chunks: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[int] = []
    cohorts_list: list[str] = []

    print(f"Found subjects: {len(subjects)}")
    print("Extracting raw rich features...")

    for subject in subjects:
        mat = _load_mat_auto(subject.path)
        if not all(key in mat for key in KEY_TO_LABEL):
            print(f"[SKIP] missing required key(s): {subject.path}")
            continue

        for key, label in KEY_TO_LABEL.items():
            eeg = _ensure_channel_first(np.asarray(mat[key]), subject.path, key)
            for trial_idx in range(N_TRIALS_PER_EMOTION):
                trial_start = trial_idx * TRIAL_SAMPLES
                trial_stop = (trial_idx + 1) * TRIAL_SAMPLES
                trial = eeg[:, trial_start:trial_stop]
                if trial.shape != (N_CHANNELS, TRIAL_SAMPLES):
                    print(f"[SKIP] bad trial shape: {subject.path}::{key} trial={trial_idx}")
                    continue

                for window_idx in range(N_WINDOWS_PER_TRIAL):
                    window_start = window_idx * WINDOW_SAMPLES
                    window_stop = (window_idx + 1) * WINDOW_SAMPLES
                    window = trial[:, window_start:window_stop]
                    if window.shape != (N_CHANNELS, WINDOW_SAMPLES):
                        print(f"[SKIP] bad window shape: {subject.path}::{key} window={window_idx}")
                        continue
                    batch.append(window.astype(np.float32, copy=False))
                    y_list.append(label)
                    groups_list.append(subject.group_id)
                    cohorts_list.append(subject.cohort)
                    if len(batch) >= BATCH_SIZE:
                        flush_batch(batch, feature_chunks, conn_chunks)

    flush_batch(batch, feature_chunks, conn_chunks)
    if not feature_chunks:
        raise RuntimeError("No raw features were extracted.")

    X_raw = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    X_conn = np.concatenate(conn_chunks, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups_list, dtype=np.int64)
    cohorts = np.asarray(cohorts_list, dtype="<U16")

    if not (len(X_raw) == len(y) == len(groups) == len(cohorts)):
        raise RuntimeError("Feature/label length mismatch.")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURE_DIR / "X_rich_raw.npy", X_raw)
    np.save(FEATURE_DIR / "X_conn_raw.npy", X_conn)
    np.save(FEATURE_DIR / "y_raw.npy", y)
    np.save(FEATURE_DIR / "groups_raw.npy", groups)
    np.save(FEATURE_DIR / "cohorts_raw.npy", cohorts)
    check_feature_matrix("X_rich_raw", X_raw)
    check_feature_matrix("X_conn_raw", X_conn)

    X_subject = np.concatenate([X_raw, subject_normalize(X_raw, groups)], axis=1).astype(
        np.float32,
        copy=False,
    )
    np.save(FEATURE_DIR / "X_rich_raw_subject.npy", X_subject)
    check_feature_matrix("X_rich_raw_subject", X_subject)

    X_subject_rank = np.concatenate(
        [X_raw, subject_normalize(X_raw, groups), subject_rank_normalize(X_raw, groups)],
        axis=1,
    ).astype(np.float32, copy=False)
    np.save(FEATURE_DIR / "X_rich_raw_subject_rank.npy", X_subject_rank)
    check_feature_matrix("X_rich_raw_subject_rank", X_subject_rank)

    X_rich_conn = np.concatenate([X_raw, X_conn], axis=1).astype(np.float32, copy=False)
    X_rich_conn_subject = np.concatenate(
        [X_rich_conn, subject_normalize(X_rich_conn, groups)],
        axis=1,
    ).astype(np.float32, copy=False)
    np.save(FEATURE_DIR / "X_rich_conn_raw_subject.npy", X_rich_conn_subject)
    check_feature_matrix("X_rich_conn_raw_subject", X_rich_conn_subject)

    z_path = FEATURE_DIR / "X_rich.npy"
    if z_path.exists():
        X_z = np.load(z_path)
        if len(X_z) == len(X_raw):
            X_combo = np.concatenate([X_z, X_raw], axis=1).astype(np.float32, copy=False)
            np.save(FEATURE_DIR / "X_rich_combo.npy", X_combo)
            check_feature_matrix("X_rich_combo", X_combo)
        else:
            print(f"[WARN] {z_path} length does not match raw features; combo not saved.")

    print("Saved:")
    print(FEATURE_DIR / "X_rich_raw.npy")
    print(FEATURE_DIR / "X_conn_raw.npy")
    print(FEATURE_DIR / "X_rich_raw_subject.npy")
    print(FEATURE_DIR / "X_rich_raw_subject_rank.npy")
    print(FEATURE_DIR / "X_rich_conn_raw_subject.npy")
    print(FEATURE_DIR / "X_rich_combo.npy")


if __name__ == "__main__":
    main()
