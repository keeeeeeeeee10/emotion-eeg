from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from .features import trial_summary_features


MODMA_128_RELATIVE = (
    "854301_EEG_128Channels_Resting_Lanzhou_2015",
    "EEG_128channels_resting_lanzhou_2015",
)


MODMA_128_TO_CONTEST_30 = [
    22,  # FP1
    9,   # FP2
    33,  # F7
    24,  # F3
    11,  # FZ
    124, # F4
    122, # F8
    45,  # FT7
    36,  # FC3
    6,   # FCZ
    104, # FC4
    108, # FT8
    52,  # T3/T7
    42,  # C3
    55,  # CZ
    93,  # C4
    92,  # T4/T8
    58,  # TP7
    65,  # CP3
    31,  # CPZ
    80,  # CP4
    96,  # TP8
    70,  # T5/P7
    75,  # P3
    62,  # PZ
    83,  # P4
    90,  # T6/P8
    71,  # O1
    74,  # OZ
    82,  # O2
]


@dataclass(frozen=True)
class ModmaSubject:
    subject_id: str
    label: int
    label_name: str
    path: Path


def modma_128_dir(modma_root: Path) -> Path:
    return modma_root.joinpath(*MODMA_128_RELATIVE)


def _read_subject_info(path: Path) -> dict[str, str]:
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("MODMA metadata loading requires openpyxl.") from exc

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    header = [str(value) if value is not None else "" for value in next(ws.iter_rows(values_only=True))]
    subject_idx = header.index("subject id")
    type_idx = header.index("type")
    labels: dict[str, str] = {}
    for row in ws.iter_rows(values_only=True, min_row=2):
        if not row or row[subject_idx] is None or row[type_idx] not in {"MDD", "HC"}:
            continue
        labels[str(row[subject_idx])] = str(row[type_idx])
    return labels


def list_modma_128_subjects(modma_root: Path) -> list[ModmaSubject]:
    root = modma_128_dir(modma_root)
    if not root.exists():
        raise FileNotFoundError(f"MODMA 128-channel folder not found: {root}")
    info_path = root / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
    labels = _read_subject_info(info_path)
    out: list[ModmaSubject] = []
    for path in sorted(root.glob("*.mat")):
        subject_id = path.name[:8]
        if subject_id not in labels:
            continue
        label_name = labels[subject_id]
        out.append(
            ModmaSubject(
                subject_id=subject_id,
                label=1 if label_name == "MDD" else 0,
                label_name=label_name,
                path=path,
            )
        )
    if not out:
        raise FileNotFoundError(f"No MODMA 128-channel .mat files matched metadata in {root}")
    return out


def _main_mat_array(mat: dict[str, object]) -> np.ndarray:
    candidates = [
        key
        for key in mat
        if not key.startswith("__") and key not in {"samplingRate", "Impedances_0"}
    ]
    scored: list[tuple[int, str, np.ndarray]] = []
    for key in candidates:
        try:
            arr = np.asarray(mat[key])
            if not np.issubdtype(arr.dtype, np.number):
                continue
            arr = arr.astype(np.float32, copy=False)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 2:
            continue
        if arr.shape[0] in {128, 129}:
            scored.append((arr.shape[1], key, arr))
        elif arr.shape[1] in {128, 129}:
            scored.append((arr.shape[0], key, arr.T))
    if not scored:
        raise ValueError(f"No 128/129-channel EEG array found in MODMA .mat; candidates={candidates}")
    scored.sort(key=lambda item: item[0], reverse=True)
    arr = scored[0][2]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D MODMA EEG array, got {arr.shape}")
    if arr.shape[0] == 129:
        arr = arr[:128]
    elif arr.shape[1] == 129:
        arr = arr.T[:128]
    elif arr.shape[0] != 128:
        raise ValueError(f"Expected 128/129 channel MODMA array, got {arr.shape}")
    return arr


def load_modma_128_contest30(path: Path) -> tuple[np.ndarray, float]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    fs = float(np.asarray(mat["samplingRate"]).squeeze()) if "samplingRate" in mat else 250.0
    arr = _main_mat_array(mat)
    idx = np.asarray(MODMA_128_TO_CONTEST_30, dtype=np.int64) - 1
    x = arr[idx].astype(np.float32)
    return x, fs


def robust_normalize_trial(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    median = np.median(x, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    scale = q75 - q25
    scale[scale < 1e-6] = np.std(x, axis=1, keepdims=True)[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    x = (x - median) / scale
    return np.clip(x, -12.0, 12.0).astype(np.float32)


def split_signal_to_segments(
    x: np.ndarray,
    *,
    fs: float,
    target_fs: float = 250.0,
    segment_seconds: float = 10.0,
    max_segments: int | None = None,
) -> list[np.ndarray]:
    if abs(fs - target_fs) > 1e-6:
        x = resample_poly(x, int(target_fs), int(fs), axis=1).astype(np.float32)
    samples = int(round(target_fs * segment_seconds))
    n_segments = x.shape[1] // samples
    if max_segments is not None:
        n_segments = min(n_segments, max_segments)
    segments: list[np.ndarray] = []
    for idx in range(n_segments):
        start = idx * samples
        stop = start + samples
        segments.append(robust_normalize_trial(x[:, start:stop]))
    return segments


def load_modma_summary_feature_matrix(
    modma_root: Path,
    *,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
    segment_seconds: float = 10.0,
    max_segments_per_subject: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        return (
            data["x"].astype(np.float32),
            data["y"].astype(np.int64),
            data["subjects"].astype(str),
            meta,
        )

    subjects = list_modma_128_subjects(modma_root)
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    subject_parts: list[str] = []
    per_subject: dict[str, int] = {}
    for subject in subjects:
        signal, fs = load_modma_128_contest30(subject.path)
        segments = split_signal_to_segments(
            signal,
            fs=fs,
            segment_seconds=segment_seconds,
            max_segments=max_segments_per_subject,
        )
        per_subject[subject.subject_id] = len(segments)
        for segment in segments:
            x_parts.append(trial_summary_features(segment, fs=250.0))
            y_parts.append(subject.label)
            subject_parts.append(subject.subject_id)
    if not x_parts:
        raise RuntimeError("No MODMA summary features were built")

    x = np.vstack(x_parts).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    subjects_arr = np.asarray(subject_parts, dtype=str)
    meta = {
        "source": "MODMA_128channels_resting_lanzhou_2015",
        "n_subjects": int(len(subjects)),
        "n_segments": int(len(y)),
        "segment_seconds": float(segment_seconds),
        "max_segments_per_subject": None if max_segments_per_subject is None else int(max_segments_per_subject),
        "positive_label": "MDD",
        "negative_label": "HC",
        "channel_indices_1based": MODMA_128_TO_CONTEST_30,
        "per_subject_segments": per_subject,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            x=x,
            y=y,
            subjects=subjects_arr,
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )
    return x, y, subjects_arr, meta
