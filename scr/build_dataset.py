from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


FS = 250
WINDOW_SECONDS = 10
WINDOW_SAMPLES = FS * WINDOW_SECONDS  # 2500
TRIAL_SECONDS = 50
TRIAL_SAMPLES = FS * TRIAL_SECONDS  # 12500
TOTAL_SAMPLES_PER_EMOTION = 50000
N_CHANNELS = 30
N_TRIALS_PER_EMOTION = 4
N_WINDOWS_PER_TRIAL = 5

KEY_TO_LABEL = {
    "EEG_data_neu": 0,
    "EEG_data_pos": 1,
}


@dataclass
class SubjectRecord:
    path: Path
    subject_id: str
    group_id: int
    cohort: str


def _load_mat_auto(mat_path: Path) -> dict[str, Any]:
    try:
        return loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    except Exception:
        data: dict[str, Any] = {}
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                data[k] = f[k][()]
        return data


def _ensure_channel_first(arr: np.ndarray, mat_path: Path, key: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{mat_path}::{key} expected 2D, got shape={arr.shape}")

    if arr.shape == (N_CHANNELS, TOTAL_SAMPLES_PER_EMOTION):
        return arr
    if arr.shape == (TOTAL_SAMPLES_PER_EMOTION, N_CHANNELS):
        return arr.T
    raise ValueError(
        f"{mat_path}::{key} bad shape={arr.shape}, "
        f"expected {(N_CHANNELS, TOTAL_SAMPLES_PER_EMOTION)} or {(TOTAL_SAMPLES_PER_EMOTION, N_CHANNELS)}"
    )


def _zscore_per_channel(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-6)


def _cohort_from_path(mat_path: Path) -> str:
    folder = mat_path.parent.name.lower()
    if any(k in folder for k in ("normal", "正常")):
        return "normal"
    if any(k in folder for k in ("patient", "抑郁", "depress")):
        return "patient"
    return "unknown"


def iter_subjects(root: Path) -> list[SubjectRecord]:
    mat_files = sorted(root.rglob("*.mat"))
    subjects: list[SubjectRecord] = []
    for idx, p in enumerate(mat_files):
        subjects.append(
            SubjectRecord(
                path=p,
                subject_id=p.stem,
                group_id=idx,
                cohort=_cohort_from_path(p),
            )
        )
    return subjects


def build_dataset(train_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subjects = iter_subjects(train_root)
    if not subjects:
        raise FileNotFoundError(f"No mat files found under {train_root}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[int] = []
    cohorts_list: list[str] = []

    print(f"Found subjects: {len(subjects)}")
    print("Building windows...")

    for s in subjects:
        try:
            mat = _load_mat_auto(s.path)
        except Exception as e:
            print(f"[SKIP] load failed: {s.path} -> {type(e).__name__}: {e}")
            continue

        if not all(k in mat for k in KEY_TO_LABEL):
            print(f"[SKIP] missing required key(s): {s.path}")
            continue

        for key, label in KEY_TO_LABEL.items():
            eeg = np.asarray(mat[key])
            try:
                eeg = _ensure_channel_first(eeg, s.path, key)
            except ValueError as e:
                print(f"[SKIP] {e}")
                continue

            for t in range(N_TRIALS_PER_EMOTION):
                t0 = t * TRIAL_SAMPLES
                t1 = (t + 1) * TRIAL_SAMPLES
                trial = eeg[:, t0:t1]
                if trial.shape != (N_CHANNELS, TRIAL_SAMPLES):
                    print(f"[SKIP] bad trial shape: {s.path}::{key} trial={t}, shape={trial.shape}")
                    continue

                for w in range(N_WINDOWS_PER_TRIAL):
                    w0 = w * WINDOW_SAMPLES
                    w1 = (w + 1) * WINDOW_SAMPLES
                    window = trial[:, w0:w1]
                    if window.shape != (N_CHANNELS, WINDOW_SAMPLES):
                        print(
                            f"[SKIP] bad window shape: {s.path}::{key} trial={t} window={w}, "
                            f"shape={window.shape}"
                        )
                        continue

                    window = _zscore_per_channel(window).astype(np.float32, copy=False)
                    X_list.append(window)
                    y_list.append(label)
                    groups_list.append(s.group_id)
                    cohorts_list.append(s.cohort)

    if not X_list:
        raise RuntimeError("No valid windows built. Please run check_mat.py first.")

    X = np.stack(X_list, axis=0)  # [num_samples, 30, 2500]
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups_list, dtype=np.int64)
    cohorts = np.asarray(cohorts_list, dtype="<U16")
    return X, y, groups, cohorts


def save_outputs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cohorts: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_windows.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "groups.npy", groups)
    np.save(out_dir / "cohorts.npy", cohorts)


def main() -> None:
    train_root = Path("data/train")
    out_dir = Path("output/features")

    X, y, groups, cohorts = build_dataset(train_root)
    save_outputs(X, y, groups, cohorts, out_dir)

    print("\nBuild finished")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}, labels: {np.unique(y)}")
    print(f"groups shape: {groups.shape}, unique subjects: {len(np.unique(groups))}")
    print(f"cohorts shape: {cohorts.shape}, unique cohorts: {np.unique(cohorts)}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
