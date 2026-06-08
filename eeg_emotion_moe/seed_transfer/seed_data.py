from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat, whosmat

from .channels import seed_to_contest_mapping
from .features import bandpower_de_features, flatten_channel_band_features, summarize_window_features


SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1], dtype=np.int64)


@dataclass(frozen=True)
class SeedLoadConfig:
    seed_root: Path
    source: str = "raw"
    feature_dir: str = "ExtractedFeatures_1s"
    feature_name: str = "de_LDS"
    raw_dir: str = "Preprocessed_EEG"
    raw_fs: float = 200.0
    window_seconds: float = 1.0
    segment_seconds: float = 10.0
    max_sessions: int | None = None
    include_positive: bool = True
    include_neutral: bool = True
    max_windows: int | None = None
    random_seed: int = 2026


def list_seed_feature_files(seed_root: Path, feature_dir: str) -> list[Path]:
    folder = seed_root / feature_dir
    if not folder.exists():
        raise FileNotFoundError(f"SEED feature folder not found: {folder}")
    files = sorted(p for p in folder.glob("*.mat") if p.name.lower() != "label.mat")
    if not files:
        raise FileNotFoundError(f"No session .mat files found under {folder}")
    return files


def list_seed_raw_files(seed_root: Path, raw_dir: str) -> list[Path]:
    folder = seed_root / raw_dir
    if not folder.exists():
        raise FileNotFoundError(f"SEED raw/preprocessed folder not found: {folder}")
    files = sorted(p for p in folder.glob("*.mat") if p.name.lower() != "label.mat")
    if not files:
        raise FileNotFoundError(f"No SEED preprocessed .mat files found under {folder}")
    return files


def _selected_trials(include_positive: bool, include_neutral: bool) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for trial_idx, label in enumerate(SEED_LABELS, start=1):
        if label == 1 and include_positive:
            selected.append((trial_idx, 1))
        elif label == 0 and include_neutral:
            selected.append((trial_idx, 0))
    return selected


def _apply_window_limit(
    x: np.ndarray,
    y: np.ndarray,
    max_windows: int | None,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_windows is not None and max_windows > 0 and x.shape[0] > max_windows:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(x.shape[0], size=max_windows, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def load_seed_official_feature_matrix(config: SeedLoadConfig) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mapping = seed_to_contest_mapping()
    session_files = list_seed_feature_files(config.seed_root, config.feature_dir)
    if config.max_sessions is not None and config.max_sessions > 0:
        session_files = session_files[: config.max_sessions]
    selected_trials = _selected_trials(config.include_positive, config.include_neutral)
    variable_names = [f"{config.feature_name}{trial_idx}" for trial_idx, _ in selected_trials]

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    session_count = 0
    missing_vars: list[str] = []

    for mat_path in session_files:
        data = loadmat(mat_path, variable_names=variable_names)
        session_count += 1
        for trial_idx, label in selected_trials:
            var_name = f"{config.feature_name}{trial_idx}"
            if var_name not in data:
                missing_vars.append(f"{mat_path.name}:{var_name}")
                continue
            arr = np.asarray(data[var_name], dtype=np.float32)
            arr = arr[mapping.seed_indices, :, :]
            x_trial = flatten_channel_band_features(arr)
            y_trial = np.full(x_trial.shape[0], label, dtype=np.int64)
            x_parts.append(x_trial)
            y_parts.append(y_trial)

    if missing_vars:
        raise KeyError("Missing SEED feature variables: " + ", ".join(missing_vars[:10]))
    if not x_parts:
        raise RuntimeError("No SEED features loaded")

    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)

    x, y = _apply_window_limit(x, y, config.max_windows, config.random_seed)

    meta = {
        "source": "official_feature",
        "session_count": session_count,
        "feature_dir": config.feature_dir,
        "feature_name": config.feature_name,
        "channels": mapping.contest_channels,
        "seed_channels": mapping.seed_names,
        "n_features": int(x.shape[1]),
        "n_windows": int(x.shape[0]),
        "positive_windows": int((y == 1).sum()),
        "neutral_windows": int((y == 0).sum()),
    }
    return x, y, meta


def _seed_raw_trial_variables(mat_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, shape, dtype in whosmat(mat_path):
        match = re.search(r"_eeg(\d+)$", name)
        if not match:
            continue
        if len(shape) == 2 and shape[0] == 62:
            out[int(match.group(1))] = name
    if len(out) < 15:
        raise KeyError(f"Expected 15 SEED trial variables in {mat_path}, found {len(out)}")
    return out


def load_seed_raw_feature_matrix(config: SeedLoadConfig) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mapping = seed_to_contest_mapping()
    session_files = list_seed_raw_files(config.seed_root, config.raw_dir)
    if config.max_sessions is not None and config.max_sessions > 0:
        session_files = session_files[: config.max_sessions]
    selected_trials = _selected_trials(config.include_positive, config.include_neutral)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    session_count = 0

    for mat_path in session_files:
        trial_vars = _seed_raw_trial_variables(mat_path)
        variable_names = [trial_vars[trial_idx] for trial_idx, _ in selected_trials]
        data = loadmat(mat_path, variable_names=variable_names)
        session_count += 1
        for trial_idx, label in selected_trials:
            var_name = trial_vars[trial_idx]
            arr = np.asarray(data[var_name], dtype=np.float32)
            arr = arr[mapping.seed_indices, :]
            x_trial = bandpower_de_features(
                arr,
                fs=config.raw_fs,
                window_seconds=config.window_seconds,
            )
            y_trial = np.full(x_trial.shape[0], label, dtype=np.int64)
            x_parts.append(x_trial)
            y_parts.append(y_trial)

    if not x_parts:
        raise RuntimeError("No SEED raw features loaded")

    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    x, y = _apply_window_limit(x, y, config.max_windows, config.random_seed)

    meta = {
        "source": "raw_preprocessed",
        "session_count": session_count,
        "raw_dir": config.raw_dir,
        "feature_name": "raw_bandpower_de",
        "raw_fs": float(config.raw_fs),
        "window_seconds": float(config.window_seconds),
        "channels": mapping.contest_channels,
        "seed_channels": mapping.seed_names,
        "n_features": int(x.shape[1]),
        "n_windows": int(x.shape[0]),
        "positive_windows": int((y == 1).sum()),
        "neutral_windows": int((y == 0).sum()),
    }
    return x, y, meta


def load_seed_raw_summary_feature_matrix(config: SeedLoadConfig) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mapping = seed_to_contest_mapping()
    session_files = list_seed_raw_files(config.seed_root, config.raw_dir)
    if config.max_sessions is not None and config.max_sessions > 0:
        session_files = session_files[: config.max_sessions]
    selected_trials = _selected_trials(config.include_positive, config.include_neutral)
    windows_per_segment = int(round(config.segment_seconds / config.window_seconds))
    if windows_per_segment < 1:
        raise ValueError("segment_seconds must be >= window_seconds")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    session_count = 0

    for mat_path in session_files:
        trial_vars = _seed_raw_trial_variables(mat_path)
        variable_names = [trial_vars[trial_idx] for trial_idx, _ in selected_trials]
        data = loadmat(mat_path, variable_names=variable_names)
        session_count += 1
        for trial_idx, label in selected_trials:
            var_name = trial_vars[trial_idx]
            arr = np.asarray(data[var_name], dtype=np.float32)
            arr = arr[mapping.seed_indices, :]
            window_features = bandpower_de_features(
                arr,
                fs=config.raw_fs,
                window_seconds=config.window_seconds,
            )
            n_segments = window_features.shape[0] // windows_per_segment
            if n_segments < 1:
                continue
            for segment_idx in range(n_segments):
                start = segment_idx * windows_per_segment
                stop = start + windows_per_segment
                x_parts.append(summarize_window_features(window_features[start:stop]))
                y_parts.append(np.array([label], dtype=np.int64))

    if not x_parts:
        raise RuntimeError("No SEED raw summary features loaded")

    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    x, y = _apply_window_limit(x, y, config.max_windows, config.random_seed)

    meta = {
        "source": "raw_summary",
        "session_count": session_count,
        "raw_dir": config.raw_dir,
        "feature_name": "trial_summary_bandpower",
        "raw_fs": float(config.raw_fs),
        "window_seconds": float(config.window_seconds),
        "segment_seconds": float(config.segment_seconds),
        "channels": mapping.contest_channels,
        "seed_channels": mapping.seed_names,
        "n_features": int(x.shape[1]),
        "n_samples": int(x.shape[0]),
        "positive_samples": int((y == 1).sum()),
        "neutral_samples": int((y == 0).sum()),
    }
    return x, y, meta


def load_seed_feature_matrix(config: SeedLoadConfig) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if config.source in {"raw_summary", "summary"}:
        return load_seed_raw_summary_feature_matrix(config)
    if config.source == "raw":
        return load_seed_raw_feature_matrix(config)
    if config.source in {"official_feature", "official-feature", "feature"}:
        return load_seed_official_feature_matrix(config)
    raise ValueError(f"Unknown SEED source: {config.source}")
