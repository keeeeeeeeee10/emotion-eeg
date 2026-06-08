from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.signal import butter, sosfiltfilt

from seed_transfer.channels import CONTEST_CHANNELS


CONNECTIVITY_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
]

CONNECTIVITY_REGIONS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("left_frontal", ["FP1", "F7", "F3", "FT7", "FC3"]),
        ("midline_frontal", ["FZ", "FCZ"]),
        ("right_frontal", ["FP2", "F8", "F4", "FT8", "FC4"]),
        ("left_temporal", ["T3", "TP7", "T5"]),
        ("right_temporal", ["T4", "TP8", "T6"]),
        ("left_central_parietal", ["C3", "CP3", "P3"]),
        ("midline_central_parietal", ["CZ", "CPZ", "PZ"]),
        ("right_central_parietal", ["C4", "CP4", "P4"]),
        ("left_occipital", ["O1"]),
        ("midline_occipital", ["OZ"]),
        ("right_occipital", ["O2"]),
    ]
)

POWER_ASYMMETRY_PAIRS = [
    ("left_frontal", "right_frontal"),
    ("left_temporal", "right_temporal"),
    ("left_central_parietal", "right_central_parietal"),
    ("left_occipital", "right_occipital"),
]


def _region_channel_indices(
    regions: OrderedDict[str, list[str]],
    channels: list[str],
) -> list[np.ndarray]:
    channel_index = {name: idx for idx, name in enumerate(channels)}
    out: list[np.ndarray] = []
    for region, region_channels in regions.items():
        missing = [name for name in region_channels if name not in channel_index]
        if missing:
            raise KeyError(f"Region {region} contains missing channels: {missing}")
        out.append(np.asarray([channel_index[name] for name in region_channels], dtype=np.int64))
    return out


def _region_weights(region_indices: list[np.ndarray], n_channels: int) -> np.ndarray:
    weights = np.zeros((len(region_indices), n_channels), dtype=np.float32)
    for region_idx, channel_idx in enumerate(region_indices):
        weights[region_idx, channel_idx] = 1.0 / float(len(channel_idx))
    return weights


def _corr_from_signals(signals: np.ndarray) -> np.ndarray:
    centered = signals - signals.mean(axis=-1, keepdims=True)
    denom = np.sqrt(np.sum(centered * centered, axis=-1, keepdims=True))
    denom[denom < 1e-6] = 1.0
    normalized = centered / denom
    corr = np.einsum("nrt,nst->nrs", normalized, normalized, optimize=True)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def connectivity_feature_names(
    *,
    bands: list[tuple[str, float, float]] | None = None,
    regions: OrderedDict[str, list[str]] | None = None,
) -> list[str]:
    selected_bands = bands or CONNECTIVITY_BANDS
    selected_regions = regions or CONNECTIVITY_REGIONS
    region_names = list(selected_regions.keys())
    region_indices = _region_channel_indices(selected_regions, list(CONTEST_CHANNELS))
    names: list[str] = []
    for band_name, _, _ in selected_bands:
        for i, left in enumerate(region_names):
            for j in range(i + 1, len(region_names)):
                names.append(f"{band_name}:region_corr:{left}--{region_names[j]}")
        for region_name, channel_idx in zip(region_names, region_indices):
            if len(channel_idx) >= 2:
                names.append(f"{band_name}:within_corr:{region_name}")
        for region_name in region_names:
            names.append(f"{band_name}:region_logvar:{region_name}")
        for left, right in POWER_ASYMMETRY_PAIRS:
            if left in selected_regions and right in selected_regions:
                names.append(f"{band_name}:power_asym:{left}--{right}")
    return names


def connectivity_features(
    x: np.ndarray,
    *,
    fs: float = 250.0,
    bands: list[tuple[str, float, float]] | None = None,
    regions: OrderedDict[str, list[str]] | None = None,
    batch_size: int = 128,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Extract low-dimensional region connectivity features from raw trials.

    Input trials are shaped (N, channels, samples). Features are band-specific
    correlations among region-averaged signals, within-region synchrony,
    region log-variance, and left-right regional power asymmetry.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected raw trials shaped (N,C,T), got {x.shape}")
    if x.shape[1] != len(CONTEST_CHANNELS):
        raise ValueError(f"Expected {len(CONTEST_CHANNELS)} contest channels, got {x.shape[1]}")

    selected_bands = bands or CONNECTIVITY_BANDS
    selected_regions = regions or CONNECTIVITY_REGIONS
    region_names = list(selected_regions.keys())
    region_indices = _region_channel_indices(selected_regions, list(CONTEST_CHANNELS))
    weights = _region_weights(region_indices, x.shape[1])
    names = connectivity_feature_names(bands=selected_bands, regions=selected_regions)

    n_trials = x.shape[0]
    features = np.empty((n_trials, len(names)), dtype=np.float32)
    upper_region = np.triu_indices(len(region_names), k=1)
    asym_pairs = [
        (region_names.index(left), region_names.index(right))
        for left, right in POWER_ASYMMETRY_PAIRS
        if left in selected_regions and right in selected_regions
    ]

    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = x[start:end].astype(np.float32, copy=False)
        band_parts: list[np.ndarray] = []
        if verbose:
            print(f"connectivity batch {start}:{end}", flush=True)
        for band_name, low, high in selected_bands:
            sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
            filtered = sosfiltfilt(sos, batch, axis=-1).astype(np.float32)
            region_signals = np.einsum("rc,nct->nrt", weights, filtered, optimize=True)
            region_corr = _corr_from_signals(region_signals)[:, upper_region[0], upper_region[1]]

            within_parts: list[np.ndarray] = []
            for channel_idx in region_indices:
                if len(channel_idx) < 2:
                    continue
                channel_corr = _corr_from_signals(filtered[:, channel_idx, :])
                upper = np.triu_indices(len(channel_idx), k=1)
                within_parts.append(channel_corr[:, upper[0], upper[1]].mean(axis=1, keepdims=True))
            within_corr = (
                np.hstack(within_parts).astype(np.float32)
                if within_parts
                else np.empty((end - start, 0), dtype=np.float32)
            )

            region_logvar = np.log(np.mean(region_signals * region_signals, axis=-1) + 1e-8).astype(np.float32)
            asym = np.column_stack(
                [region_logvar[:, left] - region_logvar[:, right] for left, right in asym_pairs]
            ).astype(np.float32)
            band_parts.extend([region_corr, within_corr, region_logvar, asym])
        features[start:end] = np.hstack(band_parts).astype(np.float32)

    return features, names
