from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import welch


FS = 250
N_CHANNELS = 30
N_SAMPLES = 2500
EPS = 1e-8

# (name, low, high)
BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

LEFT_RIGHT_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),    # FP1-FP2
    (2, 6),    # F7-F8
    (3, 5),    # F3-F4
    (7, 11),   # FT7-FT8
    (8, 10),   # FC3-FC4
    (12, 16),  # T3-T4
    (13, 15),  # C3-C4
    (17, 21),  # TP7-TP8
    (18, 20),  # CP3-CP4
    (22, 26),  # T5-T6
    (23, 25),  # P3-P4
    (27, 29),  # O1-O2
)

REGIONS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),  # frontal/frontotemporal
    (12, 16, 17, 21, 22, 26),                 # temporal
    (13, 14, 15),                             # central
    (18, 19, 20, 23, 24, 25),                 # centro-parietal/parietal
    (27, 28, 29),                             # occipital
)

RATIO_BAND_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),  # theta / alpha
    (3, 2),  # beta / alpha
    (4, 3),  # gamma / beta
    (3, 1),  # beta / theta
    (2, 1),  # alpha / theta
)


def _validate_windows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [N, 30, 2500], got shape={X.shape}")
    if X.shape[1] != N_CHANNELS or X.shape[2] != N_SAMPLES:
        raise ValueError(f"X must be [N, {N_CHANNELS}, {N_SAMPLES}], got shape={X.shape}")
    return X.astype(np.float64, copy=False)


def _safe_cast_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def _band_power_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Iterable[tuple[str, float, float]] = BANDS,
) -> np.ndarray:
    # psd shape: [channels, freq_bins]
    band_powers = []
    for _, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            power = np.zeros(psd.shape[0], dtype=np.float64)
        else:
            # 对频带内功率谱积分，得到频带能量
            power = np.trapz(psd[:, mask], freqs[mask], axis=1)
        band_powers.append(power)
    # [n_bands, channels]
    return np.stack(band_powers, axis=0)


def extract_psd_features(X: np.ndarray) -> np.ndarray:
    """
    输入: X [N, 30, 2500]
    输出: X_psd [N, 150] = 30 通道 * 5 频段
    """
    X = _validate_windows(X)
    n_samples = X.shape[0]
    n_bands = len(BANDS)
    out = np.zeros((n_samples, N_CHANNELS * n_bands), dtype=np.float64)

    for i in range(n_samples):
        sample = X[i]  # [30, 2500]
        freqs, psd = welch(
            sample,
            fs=FS,
            nperseg=256,
            noverlap=128,
            axis=1,
            detrend="constant",
            scaling="density",
        )
        band_powers = _band_power_from_psd(freqs, psd, BANDS)  # [5, 30]
        log_powers = np.log(band_powers + EPS)
        # 按频段优先展开: ch1_delta...ch30_delta, ch1_theta...
        out[i] = log_powers.reshape(-1)

    return _safe_cast_float32(out)


def extract_de_features(X: np.ndarray) -> np.ndarray:
    """
    输入: X [N, 30, 2500]
    输出: X_de [N, 150] = 30 通道 * 5 频段

    简化 DE:
    DE = 0.5 * log(2*pi*e*band_power + eps)
    """
    X = _validate_windows(X)
    n_samples = X.shape[0]
    n_bands = len(BANDS)
    out = np.zeros((n_samples, N_CHANNELS * n_bands), dtype=np.float64)

    const = 2.0 * np.pi * np.e
    for i in range(n_samples):
        sample = X[i]  # [30, 2500]
        freqs, psd = welch(
            sample,
            fs=FS,
            nperseg=256,
            noverlap=128,
            axis=1,
            detrend="constant",
            scaling="density",
        )
        band_powers = _band_power_from_psd(freqs, psd, BANDS)  # [5, 30]
        de = 0.5 * np.log(const * band_powers + EPS)
        out[i] = de.reshape(-1)

    return _safe_cast_float32(out)


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """
    可选统计特征 baseline:
    每通道 4 个统计量: mean/std/min/max -> 30*4=120 维
    """
    X = _validate_windows(X)
    mean = X.mean(axis=2)
    std = X.std(axis=2)
    min_v = X.min(axis=2)
    max_v = X.max(axis=2)
    feats = np.concatenate([mean, std, min_v, max_v], axis=1)
    return _safe_cast_float32(feats)


def extract_connectivity_features(X: np.ndarray) -> np.ndarray:
    """
    Channel-connectivity descriptors for each 10-second window.

    Uses upper-triangular channel correlations from the signal and its first
    derivative. This captures spatial synchronization patterns that complement
    per-channel spectral features.
    """
    X = _validate_windows(X)
    iu = np.triu_indices(N_CHANNELS, k=1)
    out = np.zeros((X.shape[0], len(iu[0]) * 2), dtype=np.float64)

    for i, sample in enumerate(X):
        sample_z = (sample - sample.mean(axis=1, keepdims=True)) / (
            sample.std(axis=1, keepdims=True) + EPS
        )
        corr = np.corrcoef(sample_z)

        diff = np.diff(sample_z, axis=1)
        diff = (diff - diff.mean(axis=1, keepdims=True)) / (diff.std(axis=1, keepdims=True) + EPS)
        diff_corr = np.corrcoef(diff)

        out[i] = np.concatenate([corr[iu], diff_corr[iu]], axis=0)

    return _safe_cast_float32(out)


def extract_rich_features(X: np.ndarray) -> np.ndarray:
    """
    Cross-subject EEG features for each 10-second window.

    The feature set keeps the original band power/DE signal and adds
    normalization-heavy descriptors that tend to transfer better across people:
    relative band power, band ratios, spectral entropy, Hjorth dynamics,
    left-right asymmetry, and coarse brain-region summaries.
    """
    X = _validate_windows(X)
    n_samples = X.shape[0]
    n_bands = len(BANDS)
    total_band = (1.0, 45.0)
    const = 2.0 * np.pi * np.e

    feature_blocks: list[np.ndarray] = []
    for i in range(n_samples):
        sample = X[i]  # [30, 2500]
        freqs, psd = welch(
            sample,
            fs=FS,
            nperseg=512,
            noverlap=256,
            axis=1,
            detrend="constant",
            scaling="density",
        )

        band_power = _band_power_from_psd(freqs, psd, BANDS)  # [5, 30]
        log_power = np.log(band_power + EPS)
        de = 0.5 * np.log(const * band_power + EPS)

        total_mask = (freqs >= total_band[0]) & (freqs < total_band[1])
        total_power = np.trapz(psd[:, total_mask], freqs[total_mask], axis=1) + EPS
        rel_power = band_power / total_power[None, :]

        ratio_features = []
        for a_idx, b_idx in RATIO_BAND_PAIRS:
            ratio_features.append(np.log((band_power[a_idx] + EPS) / (band_power[b_idx] + EPS)))
        band_ratios = np.stack(ratio_features, axis=0)  # [n_ratios, 30]

        psd_total = psd[:, total_mask]
        psd_prob = psd_total / (psd_total.sum(axis=1, keepdims=True) + EPS)
        spectral_entropy = -np.sum(psd_prob * np.log(psd_prob + EPS), axis=1)
        spectral_entropy /= np.log(psd_prob.shape[1] + EPS)
        peak_freq = freqs[total_mask][np.argmax(psd_total, axis=1)] / total_band[1]

        dx = np.diff(sample, axis=1)
        ddx = np.diff(dx, axis=1)
        var_x = np.var(sample, axis=1) + EPS
        var_dx = np.var(dx, axis=1) + EPS
        var_ddx = np.var(ddx, axis=1) + EPS
        mobility = np.sqrt(var_dx / var_x)
        complexity = np.sqrt(var_ddx / var_dx) / (mobility + EPS)
        line_length = np.mean(np.abs(dx), axis=1)
        zero_crossing = np.mean(sample[:, :-1] * sample[:, 1:] < 0, axis=1)
        hjorth_like = np.stack(
            [mobility, complexity, line_length, zero_crossing],
            axis=0,
        )  # [4, 30]

        left = np.asarray([p[0] for p in LEFT_RIGHT_PAIRS], dtype=np.int64)
        right = np.asarray([p[1] for p in LEFT_RIGHT_PAIRS], dtype=np.int64)
        dasm = log_power[:, left] - log_power[:, right]
        rasm = np.log((rel_power[:, left] + EPS) / (rel_power[:, right] + EPS))
        asym = np.concatenate([dasm, rasm, np.abs(dasm)], axis=0)

        region_log = np.stack([log_power[:, idx].mean(axis=1) for idx in REGIONS], axis=0)
        region_rel = np.stack([rel_power[:, idx].mean(axis=1) for idx in REGIONS], axis=0)
        region_summary = np.concatenate([region_log, region_rel], axis=0)

        blocks = [
            log_power.reshape(-1),
            de.reshape(-1),
            rel_power.reshape(-1),
            band_ratios.reshape(-1),
            spectral_entropy.reshape(-1),
            peak_freq.reshape(-1),
            hjorth_like.reshape(-1),
            asym.reshape(-1),
            region_summary.reshape(-1),
        ]
        feature_blocks.append(np.concatenate(blocks, axis=0))

    return _safe_cast_float32(np.stack(feature_blocks, axis=0))


def check_feature_matrix(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"nan={np.isnan(arr).sum()}, inf={np.isinf(arr).sum()}, "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}"
    )
