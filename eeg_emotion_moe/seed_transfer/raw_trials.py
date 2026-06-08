from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly

from .channels import seed_to_contest_mapping
from .contest_data import load_contest_training_trials, load_public_test_trials
from .paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEED_ROOT, ensure_dir
from .seed_data import SEED_LABELS, list_seed_raw_files


TARGET_FS = 250
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS


def _seed_trial_variables(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, shape, _ in whosmat(path):
        match = re.search(r"_eeg(\d+)$", name)
        if match and len(shape) == 2 and shape[0] == 62:
            out[int(match.group(1))] = name
    if len(out) < 15:
        raise RuntimeError(f"Expected 15 SEED trial variables in {path}, found {len(out)}")
    return out


def normalize_trials(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def build_seed_raw_trial_cache(
    *,
    seed_root: Path = DEFAULT_SEED_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    refresh: bool = False,
) -> Path:
    output_dir = ensure_dir(output_dir)
    cache_path = output_dir / "cache_seed_raw_trials_30x2500.npz"
    if cache_path.exists() and not refresh:
        return cache_path

    mapping = seed_to_contest_mapping()
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    subjects: list[str] = []
    sessions: list[str] = []
    trial_ids: list[int] = []

    for path in list_seed_raw_files(seed_root, "Preprocessed_EEG"):
        subject = path.stem.split("_", 1)[0]
        trial_vars = _seed_trial_variables(path)
        selected = [(idx, int(label)) for idx, label in enumerate(SEED_LABELS, start=1) if int(label) in (0, 1)]
        mat = loadmat(path, variable_names=[trial_vars[idx] for idx, _ in selected])
        for trial_idx, label in selected:
            arr = np.asarray(mat[trial_vars[trial_idx]], dtype=np.float32)[mapping.seed_indices, :]
            arr_250 = resample_poly(arr, up=5, down=4, axis=-1).astype(np.float32)
            n_segments = arr_250.shape[1] // TARGET_SAMPLES
            for segment_idx in range(n_segments):
                start = segment_idx * TARGET_SAMPLES
                stop = start + TARGET_SAMPLES
                x_parts.append(arr_250[:, start:stop])
                y_parts.append(label)
                subjects.append(subject)
                sessions.append(path.stem)
                trial_ids.append(trial_idx)

    x = normalize_trials(np.stack(x_parts).astype(np.float32))
    y = np.asarray(y_parts, dtype=np.int64)
    meta = {
        "source": "SEED Preprocessed_EEG positive/neutral only",
        "shape": list(x.shape),
        "fs": TARGET_FS,
        "seconds": TARGET_SECONDS,
        "subjects": len(set(subjects)),
        "positive": int((y == 1).sum()),
        "neutral": int((y == 0).sum()),
    }
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        subjects=np.asarray(subjects, dtype=str),
        sessions=np.asarray(sessions, dtype=str),
        trial_ids=np.asarray(trial_ids, dtype=np.int64),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return cache_path


def build_contest_raw_trial_cache(
    *,
    contest_root: Path = DEFAULT_CONTEST_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    refresh: bool = False,
) -> Path:
    output_dir = ensure_dir(output_dir)
    cache_path = output_dir / "cache_contest_train_raw_trials_30x2500.npz"
    if cache_path.exists() and not refresh:
        return cache_path

    trials = load_contest_training_trials(contest_root)
    x = normalize_trials(np.stack([trial.x.astype(np.float32) for trial in trials]))
    y = np.asarray([int(trial.y) for trial in trials], dtype=np.int64)
    subjects = np.asarray([trial.user_id for trial in trials], dtype=str)
    trial_ids = np.asarray([trial.trial_id for trial in trials], dtype=np.int64)
    meta = {
        "source": "contest training set",
        "shape": list(x.shape),
        "fs": TARGET_FS,
        "seconds": TARGET_SECONDS,
        "subjects": len(set(subjects.tolist())),
        "positive": int((y == 1).sum()),
        "neutral": int((y == 0).sum()),
    }
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        subjects=subjects,
        trial_ids=trial_ids,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return cache_path


def build_public_raw_trial_cache(
    *,
    contest_root: Path = DEFAULT_CONTEST_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    refresh: bool = False,
) -> Path:
    output_dir = ensure_dir(output_dir)
    cache_path = output_dir / "cache_public_raw_trials_30x2500.npz"
    if cache_path.exists() and not refresh:
        return cache_path

    trials = load_public_test_trials(contest_root)
    x = normalize_trials(np.stack([trial.x.astype(np.float32) for trial in trials]))
    subjects = np.asarray([trial.user_id for trial in trials], dtype=str)
    trial_ids = np.asarray([trial.trial_id for trial in trials], dtype=np.int64)
    meta = {
        "source": "contest public test set",
        "shape": list(x.shape),
        "fs": TARGET_FS,
        "seconds": TARGET_SECONDS,
        "subjects": len(set(subjects.tolist())),
    }
    np.savez_compressed(
        cache_path,
        x=x,
        subjects=subjects,
        trial_ids=trial_ids,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return cache_path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}

