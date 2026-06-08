from __future__ import annotations

import numpy as np
from scipy.signal import welch


BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
]

CONTEST_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]

ASYMMETRY_PAIRS = [
    ("FP1", "FP2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FT7", "FT8"),
    ("FC3", "FC4"),
    ("T3", "T4"),
    ("C3", "C4"),
    ("TP7", "TP8"),
    ("CP3", "CP4"),
    ("T5", "T6"),
    ("P3", "P4"),
    ("O1", "O2"),
]

FRONTAL_CHANNELS = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"]
POSTERIOR_CHANNELS = ["T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]


def flatten_channel_band_features(features: np.ndarray) -> np.ndarray:
    """Convert (channels, windows, bands) to (windows, channels * bands)."""
    if features.ndim != 3:
        raise ValueError(f"Expected 3D feature array, got {features.shape}")
    return np.transpose(features, (1, 0, 2)).reshape(features.shape[1], -1).astype(np.float32)


def segment_signal(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Return non-overlapping windows as (n_windows, channels, window_samples)."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {x.shape}")
    if x.shape[0] < x.shape[1]:
        channels_first = x
    else:
        channels_first = x.T
    n_channels, n_samples = channels_first.shape
    n_windows = n_samples // window_samples
    if n_windows < 1:
        raise ValueError(f"Signal has {n_samples} samples, less than one window of {window_samples}")
    trimmed = channels_first[:, : n_windows * window_samples]
    return trimmed.reshape(n_channels, n_windows, window_samples).transpose(1, 0, 2).astype(np.float32)


def bandpower_de_features(
    signal: np.ndarray,
    *,
    fs: float = 250.0,
    window_seconds: float = 1.0,
    bands: list[tuple[str, float, float]] | None = None,
) -> np.ndarray:
    """Extract log bandpower features from raw EEG.

    Returns an array with shape (n_windows, channels * bands).
    """
    selected_bands = bands or BANDS
    window_samples = int(round(fs * window_seconds))
    windows = segment_signal(signal, window_samples)
    n_windows, n_channels, n_samples = windows.shape
    nperseg = min(n_samples, 256)
    freqs, psd = welch(windows, fs=fs, nperseg=nperseg, axis=-1)
    out = np.empty((n_windows, n_channels, len(selected_bands)), dtype=np.float32)
    for band_idx, (_, low, high) in enumerate(selected_bands):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            out[:, :, band_idx] = 0.0
            continue
        power = np.trapezoid(psd[:, :, mask], freqs[mask], axis=-1)
        out[:, :, band_idx] = np.log(power + 1e-8)
    return out.reshape(n_windows, -1).astype(np.float32)


def aggregate_window_probabilities(probabilities: np.ndarray) -> float:
    if probabilities.size == 0:
        raise ValueError("Cannot aggregate empty probabilities")
    return float(np.mean(probabilities))


def summarize_window_features(window_features: np.ndarray, *, n_channels: int = 30, n_bands: int = 5) -> np.ndarray:
    """Summarize 1-second window features into one trial-level vector."""
    if window_features.ndim != 2:
        raise ValueError(f"Expected 2D window feature matrix, got {window_features.shape}")
    if window_features.shape[1] != n_channels * n_bands:
        raise ValueError(
            f"Expected {n_channels * n_bands} features, got {window_features.shape[1]}"
        )
    cube = window_features.reshape(window_features.shape[0], n_channels, n_bands)
    mean_cb = cube.mean(axis=0)
    std_cb = cube.std(axis=0)
    median_cb = np.median(cube, axis=0)

    channel_index = {name: idx for idx, name in enumerate(CONTEST_CHANNELS)}
    asym_diff: list[np.ndarray] = []
    asym_ratio: list[np.ndarray] = []
    for left, right in ASYMMETRY_PAIRS:
        left_values = mean_cb[channel_index[left]]
        right_values = mean_cb[channel_index[right]]
        diff = left_values - right_values
        ratio = diff / (np.abs(left_values) + np.abs(right_values) + 1e-6)
        asym_diff.append(diff)
        asym_ratio.append(ratio)

    frontal_idx = [channel_index[name] for name in FRONTAL_CHANNELS]
    posterior_idx = [channel_index[name] for name in POSTERIOR_CHANNELS]
    frontal_posterior = mean_cb[frontal_idx].mean(axis=0) - mean_cb[posterior_idx].mean(axis=0)
    global_mean = mean_cb.mean(axis=0)
    global_std = mean_cb.std(axis=0)
    temporal_global_std = std_cb.mean(axis=0)

    parts = [
        mean_cb.ravel(),
        std_cb.ravel(),
        median_cb.ravel(),
        np.concatenate(asym_diff),
        np.concatenate(asym_ratio),
        frontal_posterior,
        global_mean,
        global_std,
        temporal_global_std,
    ]
    return np.concatenate(parts).astype(np.float32)


def trial_summary_features(
    signal: np.ndarray,
    *,
    fs: float = 250.0,
    window_seconds: float = 1.0,
) -> np.ndarray:
    window_features = bandpower_de_features(signal, fs=fs, window_seconds=window_seconds)
    return summarize_window_features(window_features)
