from __future__ import annotations

from pathlib import Path

import numpy as np

from features import check_feature_matrix


FEATURE_DIR = Path("output/features")
WINDOWS_PER_TRIAL = 5
TRIALS_PER_SUBJECT = 8
POSITIVES_PER_SUBJECT = 4


def subject_rank_8(X: np.ndarray) -> np.ndarray:
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order, np.arange(X.shape[1])] = np.arange(len(X), dtype=np.float32)[:, None]
    return 2.0 * (ranks / float(len(X) - 1)) - 1.0


def build_pseudo_features(
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cohorts: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    X_subject_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    cv_group_list: list[np.ndarray] = []
    rank_group_list: list[np.ndarray] = []
    cohort_list: list[np.ndarray] = []

    pseudo_group_id = 0
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) != TRIALS_PER_SUBJECT * WINDOWS_PER_TRIAL:
            raise ValueError(f"Subject {group} expected 40 windows, got {len(idx)}")

        for window_pos in range(WINDOWS_PER_TRIAL):
            selected = idx[np.arange(TRIALS_PER_SUBJECT) * WINDOWS_PER_TRIAL + window_pos]
            X8 = X_raw[selected]
            y8 = y[selected]
            if int(y8.sum()) != POSITIVES_PER_SUBJECT:
                raise ValueError(f"Pseudo group {pseudo_group_id} labels are not 4/8 positive.")

            X_norm = (X8 - X8.mean(axis=0, keepdims=True)) / (X8.std(axis=0, keepdims=True) + 1e-6)
            X_rank = subject_rank_8(X8)
            X_subject_list.append(
                np.concatenate([X8, X_norm], axis=1).astype(np.float32, copy=False)
            )
            X_list.append(np.concatenate([X8, X_norm, X_rank], axis=1).astype(np.float32, copy=False))
            y_list.append(y8.astype(np.int64, copy=False))
            cv_group_list.append(np.full(TRIALS_PER_SUBJECT, int(group), dtype=np.int64))
            rank_group_list.append(np.full(TRIALS_PER_SUBJECT, pseudo_group_id, dtype=np.int64))
            if cohorts is not None:
                cohort_list.append(np.full(TRIALS_PER_SUBJECT, cohorts[idx[0]], dtype="<U16"))
            pseudo_group_id += 1

    X_subject_out = np.concatenate(X_subject_list, axis=0).astype(np.float32, copy=False)
    X_out = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    y_out = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)
    cv_groups = np.concatenate(cv_group_list, axis=0).astype(np.int64, copy=False)
    rank_groups = np.concatenate(rank_group_list, axis=0).astype(np.int64, copy=False)
    cohorts_out = np.concatenate(cohort_list, axis=0) if cohort_list else None
    return X_subject_out, X_out, y_out, cv_groups, rank_groups, cohorts_out


def main() -> None:
    x_path = FEATURE_DIR / "X_rich_raw.npy"
    y_path = FEATURE_DIR / "y.npy"
    group_path = FEATURE_DIR / "groups.npy"
    cohort_path = FEATURE_DIR / "cohorts.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing {x_path}; run scr/make_raw_features.py first.")

    X_raw = np.load(x_path)
    y = np.load(y_path).astype(np.int64, copy=False)
    groups = np.load(group_path).astype(np.int64, copy=False)
    cohorts = np.load(cohort_path) if cohort_path.exists() else None

    X_subject_out, X_out, y_out, cv_groups, rank_groups, cohorts_out = build_pseudo_features(
        X_raw,
        y,
        groups,
        cohorts,
    )

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURE_DIR / "X_rich_pseudo_subject.npy", X_subject_out)
    np.save(FEATURE_DIR / "X_rich_pseudo_subject_rank.npy", X_out)
    np.save(FEATURE_DIR / "y_pseudo.npy", y_out)
    np.save(FEATURE_DIR / "groups_pseudo_cv.npy", cv_groups)
    np.save(FEATURE_DIR / "groups_pseudo_rank.npy", rank_groups)
    if cohorts_out is not None:
        np.save(FEATURE_DIR / "cohorts_pseudo.npy", cohorts_out)

    check_feature_matrix("X_rich_pseudo_subject", X_subject_out)
    check_feature_matrix("X_rich_pseudo_subject_rank", X_out)
    print(f"pseudo rank groups: {len(np.unique(rank_groups))}")
    print(f"cv subjects: {len(np.unique(cv_groups))}")
    print("Saved:")
    print(FEATURE_DIR / "X_rich_pseudo_subject.npy")
    print(FEATURE_DIR / "X_rich_pseudo_subject_rank.npy")


if __name__ == "__main__":
    main()
