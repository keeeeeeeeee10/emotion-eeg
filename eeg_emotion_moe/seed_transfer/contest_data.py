from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .features import bandpower_de_features, trial_summary_features


@dataclass(frozen=True)
class ContestTrial:
    user_id: str
    trial_id: int
    x: np.ndarray
    y: int | None
    source_file: Path


def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading contest training .mat files requires h5py because they are MATLAB v7.3/HDF5 files. "
            "Install it in UltraSoundLab with: python -m pip install h5py"
        ) from exc
    return h5py


def _read_hdf5_dataset(path: Path, key: str) -> np.ndarray:
    h5py = _import_h5py()
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{key} not found in {path}")
        arr = np.asarray(f[key][()], dtype=np.float32)
    if arr.shape == (50000, 30):
        arr = arr.T
    if arr.shape != (30, 50000):
        raise ValueError(f"Unexpected training array shape for {path}:{key}: {arr.shape}")
    return arr


def list_contest_training_files(contest_root: Path) -> list[Path]:
    train_root = contest_root / "训练集"
    if not train_root.exists():
        raise FileNotFoundError(f"Contest training folder not found: {train_root}")
    files = sorted(train_root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No contest training .mat files found under {train_root}")
    return files


def load_contest_training_trials(contest_root: Path) -> list[ContestTrial]:
    trials: list[ContestTrial] = []
    for path in list_contest_training_files(contest_root):
        user_id = path.stem.replace("timedata", "")
        for key, label in [("EEG_data_neu", 0), ("EEG_data_pos", 1)]:
            data = _read_hdf5_dataset(path, key)
            # 4 videos * 50 seconds each. Each video becomes five 10-second trials.
            for video_idx in range(4):
                video_start = video_idx * 12500
                for segment_idx in range(5):
                    start = video_start + segment_idx * 2500
                    stop = start + 2500
                    trial_id = 1 + (0 if label == 0 else 20) + video_idx * 5 + segment_idx
                    trials.append(
                        ContestTrial(
                            user_id=user_id,
                            trial_id=trial_id,
                            x=data[:, start:stop].copy(),
                            y=label,
                            source_file=path,
                        )
                    )
    return trials


def list_public_test_files(contest_root: Path) -> list[Path]:
    test_root = contest_root / "公开测试集"
    if not test_root.exists():
        raise FileNotFoundError(f"Public test folder not found: {test_root}")

    def key(path: Path) -> int:
        return int(path.stem.replace("P_test", ""))

    files = sorted(test_root.glob("P_test*.mat"), key=key)
    if not files:
        raise FileNotFoundError(f"No public test .mat files found under {test_root}")
    return files


def load_public_test_trials(contest_root: Path) -> list[ContestTrial]:
    trials: list[ContestTrial] = []
    for path in list_public_test_files(contest_root):
        mat = loadmat(path)
        if "test_eeg_c" not in mat:
            raise KeyError(f"test_eeg_c not found in {path}")
        data = np.asarray(mat["test_eeg_c"], dtype=np.float32)
        if data.shape != (30, 20000):
            raise ValueError(f"Unexpected public test shape for {path}: {data.shape}")
        for idx in range(8):
            start = idx * 2500
            stop = start + 2500
            trials.append(
                ContestTrial(
                    user_id=path.stem,
                    trial_id=idx + 1,
                    x=data[:, start:stop].copy(),
                    y=None,
                    source_file=path,
                )
            )
    return trials


def trials_to_feature_matrix(
    trials: list[ContestTrial],
    *,
    fs: float = 250.0,
    window_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None, list[tuple[str, int, int, int]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    spans: list[tuple[str, int, int, int]] = []
    start = 0
    for trial in trials:
        features = bandpower_de_features(trial.x, fs=fs, window_seconds=window_seconds)
        stop = start + features.shape[0]
        spans.append((trial.user_id, trial.trial_id, start, stop))
        x_parts.append(features)
        if trial.y is not None:
            y_parts.append(np.full(features.shape[0], trial.y, dtype=np.int64))
        start = stop
    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64) if y_parts else None
    return x, y, spans


def trials_to_summary_feature_matrix(
    trials: list[ContestTrial],
    *,
    fs: float = 250.0,
    window_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None, list[tuple[str, int, int, int]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    spans: list[tuple[str, int, int, int]] = []
    for idx, trial in enumerate(trials):
        features = trial_summary_features(trial.x, fs=fs, window_seconds=window_seconds)
        spans.append((trial.user_id, trial.trial_id, idx, idx + 1))
        x_parts.append(features)
        if trial.y is not None:
            y_parts.append(int(trial.y))
    x = np.vstack(x_parts).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64) if y_parts else None
    return x, y, spans


def aggregate_trial_predictions(
    spans: list[tuple[str, int, int, int]],
    probabilities: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, (user_id, trial_id, start, stop) in enumerate(spans):
        prob = float(np.mean(probabilities[start:stop]))
        row: dict[str, object] = {
            "user_id": user_id,
            "trial_id": int(trial_id),
            "probability": prob,
            "Emotion_label": int(prob >= threshold),
        }
        if labels is not None:
            row["true_label"] = int(labels[start])
        rows.append(row)
    return rows
