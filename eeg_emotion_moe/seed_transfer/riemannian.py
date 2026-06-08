from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


RIEMANN_BANDS = [
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
]


def _logm_spd(cov: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-6, None)
    return (vectors * np.log(values)) @ vectors.T


def _upper_triangular(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0])
    return mat[idx]


def riemann_features(
    x: np.ndarray,
    *,
    fs: float = 250.0,
    bands: list[tuple[str, float, float]] | None = None,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Log-covariance features for trials shaped (N, channels, samples)."""
    selected_bands = bands or RIEMANN_BANDS
    if x.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {x.shape}")
    feats: list[np.ndarray] = []
    eye = np.eye(x.shape[1], dtype=np.float32)
    for trial in x.astype(np.float32):
        parts: list[np.ndarray] = []
        for _, low, high in selected_bands:
            sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
            filtered = sosfiltfilt(sos, trial, axis=-1).astype(np.float32)
            cov = np.cov(filtered)
            trace = float(np.trace(cov))
            cov = cov / max(trace, 1e-6)
            cov = (1.0 - shrinkage) * cov + shrinkage * eye / eye.shape[0]
            log_cov = _logm_spd(cov)
            parts.append(_upper_triangular(log_cov))
        feats.append(np.concatenate(parts))
    return np.vstack(feats).astype(np.float32)

