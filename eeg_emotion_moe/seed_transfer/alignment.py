from __future__ import annotations

import numpy as np


def inv_sqrtm_spd(matrix: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix.astype(np.float64))
    values = np.clip(values, eps, None)
    return ((vectors * (1.0 / np.sqrt(values))) @ vectors.T).astype(np.float32)


def trial_covariance(
    trial: np.ndarray,
    *,
    trace_normalize: bool = True,
    shrinkage: float = 0.05,
) -> np.ndarray:
    centered = trial - trial.mean(axis=-1, keepdims=True)
    cov = centered @ centered.T / max(centered.shape[-1] - 1, 1)
    if trace_normalize:
        cov = cov / max(float(np.trace(cov)), 1e-6)
    if shrinkage > 0:
        eye = np.eye(cov.shape[0], dtype=np.float32)
        cov = (1.0 - shrinkage) * cov + shrinkage * eye / cov.shape[0]
    return cov.astype(np.float32)


def subject_reference_covariance(
    x_subject: np.ndarray,
    *,
    trace_normalize: bool = True,
    shrinkage: float = 0.05,
) -> np.ndarray:
    cov = np.mean(
        np.stack(
            [
                trial_covariance(
                    trial,
                    trace_normalize=trace_normalize,
                    shrinkage=shrinkage,
                )
                for trial in x_subject
            ],
            axis=0,
        ),
        axis=0,
    )
    cov = 0.5 * (cov + cov.T)
    return cov.astype(np.float32)


def euclidean_align_trials(
    x: np.ndarray,
    subjects: np.ndarray,
    *,
    trace_normalize: bool = True,
    shrinkage: float = 0.05,
) -> np.ndarray:
    """Apply per-subject Euclidean Alignment to raw EEG trials.

    The alignment matrix is estimated from all trials available for a subject.
    In contest/public inference this is an unsupervised subject-level transform.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected raw trials shaped (N,C,T), got {x.shape}")
    out = np.empty_like(x, dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        reference = subject_reference_covariance(
            x[idx],
            trace_normalize=trace_normalize,
            shrinkage=shrinkage,
        )
        transform = inv_sqrtm_spd(reference)
        out[idx] = np.einsum("cd,ndt->nct", transform, x[idx], optimize=True).astype(np.float32)
    return out
