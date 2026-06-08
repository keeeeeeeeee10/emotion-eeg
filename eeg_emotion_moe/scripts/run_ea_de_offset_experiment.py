from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.alignment import euclidean_align_trials
from seed_transfer.features import bandpower_de_features, summarize_window_features
from seed_transfer.model import binary_metrics
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Literature-inspired EA + multi-window DE feature experiment with nested "
            "subject-wise offset/top4 validation."
        )
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--outer-seeds", type=str, default="2026,2031,2042")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--offsets", type=str, default="1,3,4")
    parser.add_argument("--public-offset-weights", type=str, default="0.4,0.4,0.2")
    parser.add_argument("--feature-sets", type=str, default="summary,ea_summary,multi,ea_multi")
    parser.add_argument("--norm-modes", type=str, default="subject_center,subject_zscore")
    parser.add_argument("--reducers", type=str, default="select160,select320,pca64")
    parser.add_argument("--models", type=str, default="logreg_c0.03,logreg_c0.1,extra_trees")
    parser.add_argument("--dep-weights", type=str, default="1.0,2.0")
    parser.add_argument("--window-seconds", type=str, default="1,2,4")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--topk-video", type=int, default=4)
    return parser.parse_args()


def require_sklearn():
    try:
        from sklearn.decomposition import PCA
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn before running this script.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "PCA": PCA,
        "SelectKBest": SelectKBest,
        "StandardScaler": StandardScaler,
        "f_classif": f_classif,
        "make_pipeline": make_pipeline,
    }


def is_dep_subjects(subjects: np.ndarray) -> np.ndarray:
    return np.asarray([str(subject).upper().startswith("DEP") for subject in subjects], dtype=bool)


def subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=str)
    dep = unique[is_dep_subjects(unique)]
    hc = unique[~is_dep_subjects(unique)]
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for group in [hc, dep]:
        for idx, subject in enumerate(group):
            folds[idx % n_folds].append(str(subject))
    return [np.asarray(sorted(fold), dtype=str) for fold in folds]


def video_and_offset(trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    video_id = np.where(trial_ids <= 20, (trial_ids - 1) // 5 + 1, (trial_ids - 21) // 5 + 5).astype(np.int64)
    offset = np.where(trial_ids <= 20, (trial_ids - 1) % 5, (trial_ids - 21) % 5).astype(np.int64)
    return video_id, offset


def topk_by_subject(scores: np.ndarray, subjects: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        pred[order[:k]] = 1
    return pred


def rank_by_subject(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        if len(idx) == 1:
            out[idx] = 0.5
            continue
        order = np.argsort(scores[idx])
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        out[idx] = ranks
    return out


def score_with_groups(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> dict[str, float]:
    pred = topk_by_subject(scores, subjects, k)
    dep_mask = is_dep_subjects(subjects)
    metrics = binary_metrics(y, pred, scores)
    metrics["hc_accuracy"] = float((pred[~dep_mask] == y[~dep_mask]).mean())
    metrics["dep_accuracy"] = float((pred[dep_mask] == y[dep_mask]).mean())
    return metrics


def sample_weights(y: np.ndarray, subjects: np.ndarray, dep_weight: float) -> np.ndarray:
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    weights = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg)).astype(np.float32)
    weights[is_dep_subjects(subjects)] *= float(dep_weight)
    weights /= max(float(weights.mean()), 1e-6)
    return weights.astype(np.float32)


def normalize_by_subject(x: np.ndarray, subjects: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32, copy=True)
    out = np.empty_like(x, dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        block = x[idx]
        mean = block.mean(axis=0, keepdims=True)
        if mode == "subject_center":
            out[idx] = block - mean
        elif mode == "subject_zscore":
            scale = block.std(axis=0, keepdims=True)
            scale[scale < 1e-6] = 1.0
            out[idx] = (block - mean) / scale
        else:
            raise ValueError(f"Unknown norm mode: {mode}")
    return out.astype(np.float32)


def offset_dataset(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    *,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    video_ids, offsets = video_and_offset(trial_ids)
    mask = offsets == int(offset)
    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_subjects: list[str] = []
    rows_videos: list[int] = []
    for subject in sorted(set(subjects.tolist())):
        for video_id in range(1, 9):
            idx = np.where(mask & (subjects == subject) & (video_ids == video_id))[0]
            if len(idx) != 1:
                raise ValueError(f"Expected one row for {subject} video {video_id} offset {offset}, got {len(idx)}")
            rows_x.append(x[idx[0]])
            rows_y.append(int(y[idx[0]]))
            rows_subjects.append(str(subject))
            rows_videos.append(int(video_id))
    return (
        np.vstack(rows_x).astype(np.float32),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_subjects, dtype=str),
        np.asarray(rows_videos, dtype=np.int64),
    )


def window_summary(signal: np.ndarray, *, window_seconds: float) -> np.ndarray:
    windows = bandpower_de_features(signal, fs=250.0, window_seconds=window_seconds)
    base = summarize_window_features(windows)
    cube = windows.reshape(windows.shape[0], 30, 5)
    log_total = np.log(np.sum(np.exp(np.clip(cube, -40.0, 40.0)), axis=2, keepdims=True) + 1e-8)
    relative = (cube - log_total).reshape(windows.shape[0], -1)
    rel_summary = summarize_window_features(relative)
    q25 = np.quantile(cube, 0.25, axis=0).reshape(-1)
    q75 = np.quantile(cube, 0.75, axis=0).reshape(-1)
    first_half = cube[: max(1, cube.shape[0] // 2)].mean(axis=0)
    second_half = cube[cube.shape[0] // 2 :].mean(axis=0)
    drift = (second_half - first_half).reshape(-1)
    if cube.shape[0] > 1:
        t = np.linspace(-0.5, 0.5, cube.shape[0], dtype=np.float32)
        denom = float(np.sum(t * t))
        slope = np.tensordot(t, cube, axes=(0, 0)).reshape(-1) / max(denom, 1e-6)
    else:
        slope = np.zeros(150, dtype=np.float32)
    return np.concatenate([base, rel_summary, q25, q75, drift, slope]).astype(np.float32)


def time_domain_features(signal: np.ndarray) -> np.ndarray:
    x = signal.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    var0 = np.var(x, axis=1) + 1e-8
    var1 = np.var(dx, axis=1) + 1e-8
    var2 = np.var(ddx, axis=1) + 1e-8
    std = np.sqrt(var0)
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-8)
    line_length = np.mean(np.abs(dx), axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    ptp = np.ptp(x, axis=1)
    skew = np.mean((x / std[:, None]) ** 3, axis=1)
    kurt = np.mean((x / std[:, None]) ** 4, axis=1)
    zcr = np.mean((x[:, :-1] * x[:, 1:]) < 0, axis=1)
    return np.concatenate([std, mobility, complexity, line_length, rms, ptp, skew, kurt, zcr]).astype(np.float32)


def enhanced_trial_features(signal: np.ndarray, window_seconds: list[float]) -> np.ndarray:
    parts = [window_summary(signal, window_seconds=seconds) for seconds in window_seconds]
    parts.append(time_domain_features(signal))
    return np.concatenate(parts).astype(np.float32)


def build_feature_matrix(raw_x: np.ndarray, window_seconds: list[float]) -> np.ndarray:
    rows = []
    for idx, trial in enumerate(raw_x):
        if idx and idx % 200 == 0:
            print(f"  extracted {idx}/{len(raw_x)} trials", flush=True)
        rows.append(enhanced_trial_features(trial, window_seconds))
    return np.vstack(rows).astype(np.float32)


def load_raw(output_dir: Path) -> dict[str, np.ndarray]:
    train = np.load(output_dir / "cache_contest_train_raw_trials_30x2500.npz", allow_pickle=True)
    public = np.load(output_dir / "cache_public_raw_trials_30x2500.npz", allow_pickle=True)
    return {
        "raw_x": train["x"].astype(np.float32),
        "y": train["y"].astype(np.int64),
        "subjects": train["subjects"].astype(str),
        "trial_ids": train["trial_ids"].astype(np.int64),
        "public_raw_x": public["x"].astype(np.float32),
        "public_subjects": public["subjects"].astype(str),
        "public_trial_ids": public["trial_ids"].astype(np.int64),
    }


def load_or_build_features(
    output_dir: Path,
    *,
    window_seconds: list[float],
    refresh: bool,
) -> dict[str, np.ndarray]:
    raw = load_raw(output_dir)
    suffix = "_".join(str(seconds).replace(".", "p") for seconds in window_seconds)
    cache_path = output_dir / f"cache_ea_de_multifeature_ws{suffix}.npz"
    if cache_path.exists() and not refresh:
        data = np.load(cache_path, allow_pickle=True)
        out = {key: data[key] for key in data.files}
        return {key: value.astype(np.float32) if key.startswith("x") else value for key, value in out.items()}

    summary = np.load(output_dir / "cache_contest_trial_summary.npz", allow_pickle=True)
    public_summary = np.load(output_dir / "cache_public_trial_summary_from_raw.npz", allow_pickle=True)
    print("building subject-level EA raw trials", flush=True)
    ea_x = euclidean_align_trials(raw["raw_x"], raw["subjects"])
    public_ea_x = euclidean_align_trials(raw["public_raw_x"], raw["public_subjects"])
    print("building multi-window raw DE features", flush=True)
    multi = build_feature_matrix(raw["raw_x"], window_seconds)
    print("building multi-window EA DE features", flush=True)
    ea_multi = build_feature_matrix(ea_x, window_seconds)
    print("building public multi-window raw DE features", flush=True)
    public_multi = build_feature_matrix(raw["public_raw_x"], window_seconds)
    print("building public multi-window EA DE features", flush=True)
    public_ea_multi = build_feature_matrix(public_ea_x, window_seconds)
    np.savez_compressed(
        cache_path,
        summary=summary["x"].astype(np.float32),
        ea_summary=ea_multi[:, : summary["x"].shape[1]].astype(np.float32),
        multi=multi.astype(np.float32),
        ea_multi=ea_multi.astype(np.float32),
        public_summary=public_summary["x"].astype(np.float32),
        public_ea_summary=public_ea_multi[:, : public_summary["x"].shape[1]].astype(np.float32),
        public_multi=public_multi.astype(np.float32),
        public_ea_multi=public_ea_multi.astype(np.float32),
    )
    data = np.load(cache_path, allow_pickle=True)
    return {key: data[key].astype(np.float32) for key in data.files}


def make_model(model_name: str, seed: int):
    sk = require_sklearn()
    if model_name.startswith("logreg_c"):
        c_value = float(model_name.replace("logreg_c", ""))
        return sk["LogisticRegression"](C=c_value, solver="lbfgs", max_iter=5000, random_state=seed)
    if model_name == "hgb":
        return sk["HistGradientBoostingClassifier"](
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.25,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return sk["ExtraTreesClassifier"](
            n_estimators=260,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline_for_config(config: dict[str, object], seed: int, n_train: int, n_features: int):
    sk = require_sklearn()
    steps: list[object] = [sk["StandardScaler"]()]
    reducer = str(config["reducer"])
    if reducer.startswith("select"):
        k = min(int(reducer.replace("select", "")), n_features)
        steps.append(sk["SelectKBest"](score_func=sk["f_classif"], k=k))
    elif reducer.startswith("pca"):
        n_components = min(int(reducer.replace("pca", "")), n_train - 1, n_features)
        if n_components > 0:
            steps.append(sk["PCA"](n_components=n_components, random_state=seed))
    elif reducer != "none":
        raise ValueError(f"Unknown reducer: {reducer}")
    steps.append(make_model(str(config["model"]), seed))
    return sk["make_pipeline"](*steps)


def fit_predict_config(
    x_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    x_eval: np.ndarray,
    config: dict[str, object],
    *,
    seed: int,
) -> np.ndarray:
    pipe = make_pipeline_for_config(config, seed, x_train.shape[0], x_train.shape[1])
    final_step = pipe.steps[-1][0]
    weights = sample_weights(y_train, subjects_train, float(config["dep_weight"]))
    try:
        pipe.fit(x_train, y_train, **{f"{final_step}__sample_weight": weights})
    except TypeError:
        pipe.fit(x_train, y_train)
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(x_eval)[:, 1].astype(np.float32)
    raw = pipe.decision_function(x_eval).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))).astype(np.float32)


def config_name(config: dict[str, object]) -> str:
    return (
        f"{config['feature_set']}__{config['norm_mode']}__{config['reducer']}__"
        f"{config['model']}__depw{float(config['dep_weight']):g}"
    )


def select_config_inner_cv(
    matrices: dict[tuple[str, str], np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    configs: list[dict[str, object]],
    *,
    seed: int,
    inner_folds: int,
    k: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    train_subjects = subjects[train_mask]
    folds = subject_folds(train_subjects, inner_folds, seed)
    train_indices = np.where(train_mask)[0]
    for cfg_idx, config in enumerate(configs):
        scores = np.zeros(train_mask.sum(), dtype=np.float32)
        local_y = y[train_mask]
        local_subjects = subjects[train_mask]
        x_matrix = matrices[(str(config["feature_set"]), str(config["norm_mode"]))]
        for fold_idx, val_subjects in enumerate(folds):
            local_val = np.isin(train_subjects, val_subjects)
            val_idx = train_indices[local_val]
            fit_idx = train_indices[~local_val]
            scores[local_val] = fit_predict_config(
                x_matrix[fit_idx],
                y[fit_idx],
                subjects[fit_idx],
                x_matrix[val_idx],
                config,
                seed=seed + cfg_idx * 1000 + fold_idx,
            )
        metrics = score_with_groups(local_y, scores, local_subjects, k)
        rows.append({"config": config_name(config), **config, **metrics})
    result = pd.DataFrame(rows).sort_values(["accuracy", "dep_accuracy", "hc_accuracy"], ascending=False)
    selected_name = str(result.iloc[0]["config"])
    selected = next(config for config in configs if config_name(config) == selected_name)
    return selected, result


def nested_offset_eval(
    matrices: dict[tuple[str, str], np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    configs: list[dict[str, object]],
    *,
    seed: int,
    folds: int,
    inner_folds: int,
    k: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    oof = np.zeros(len(y), dtype=np.float32)
    outer_rows = []
    inner_top_rows = []
    for fold_idx, val_subjects in enumerate(subject_folds(subjects, folds, seed), start=1):
        val_mask = np.isin(subjects, val_subjects)
        train_mask = ~val_mask
        selected, inner_df = select_config_inner_cv(
            matrices,
            y,
            subjects,
            train_mask,
            configs,
            seed=seed + fold_idx * 100,
            inner_folds=inner_folds,
            k=k,
        )
        x_matrix = matrices[(str(selected["feature_set"]), str(selected["norm_mode"]))]
        scores = fit_predict_config(
            x_matrix[train_mask],
            y[train_mask],
            subjects[train_mask],
            x_matrix[val_mask],
            selected,
            seed=seed + fold_idx * 10000,
        )
        oof[val_mask] = scores
        metrics = score_with_groups(y[val_mask], scores, subjects[val_mask], k)
        outer_rows.append(
            {
                "seed": seed,
                "fold": fold_idx,
                "selected_config": config_name(selected),
                **selected,
                **metrics,
            }
        )
        top = inner_df.head(8).copy()
        top.insert(0, "fold", fold_idx)
        top.insert(0, "seed", seed)
        inner_top_rows.append(top)
    return oof, pd.DataFrame(outer_rows), pd.concat(inner_top_rows, ignore_index=True)


def full_cv_config_ranking(
    matrices: dict[tuple[str, str], np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    configs: list[dict[str, object]],
    *,
    seed: int,
    folds: int,
    k: int,
) -> pd.DataFrame:
    rows = []
    for cfg_idx, config in enumerate(configs):
        x_matrix = matrices[(str(config["feature_set"]), str(config["norm_mode"]))]
        scores = np.zeros(len(y), dtype=np.float32)
        for fold_idx, val_subjects in enumerate(subject_folds(subjects, folds, seed), start=1):
            val_mask = np.isin(subjects, val_subjects)
            train_mask = ~val_mask
            scores[val_mask] = fit_predict_config(
                x_matrix[train_mask],
                y[train_mask],
                subjects[train_mask],
                x_matrix[val_mask],
                config,
                seed=seed + cfg_idx * 1000 + fold_idx,
            )
        metrics = score_with_groups(y, scores, subjects, k)
        rows.append({"config": config_name(config), **config, **metrics})
    return pd.DataFrame(rows).sort_values(["accuracy", "dep_accuracy", "hc_accuracy"], ascending=False)


def main() -> None:
    args = parse_args()
    require_sklearn()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "main"
    report_dir = ensure_dir(output_dir / f"ea_de_offset_{safe_tag}")
    raw = load_raw(output_dir)
    window_seconds = [float(part.strip()) for part in args.window_seconds.split(",") if part.strip()]
    feature_data = load_or_build_features(output_dir, window_seconds=window_seconds, refresh=args.refresh_features)

    feature_sets = [part.strip() for part in args.feature_sets.split(",") if part.strip()]
    norm_modes = [part.strip() for part in args.norm_modes.split(",") if part.strip()]
    reducers = [part.strip() for part in args.reducers.split(",") if part.strip()]
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    dep_weights = [float(part.strip()) for part in args.dep_weights.split(",") if part.strip()]
    seeds = [int(part.strip()) for part in args.outer_seeds.split(",") if part.strip()]
    offsets = [int(part.strip()) for part in args.offsets.split(",") if part.strip()]
    offset_weights = np.asarray([float(part.strip()) for part in args.public_offset_weights.split(",") if part.strip()], dtype=np.float32)
    if len(offset_weights) == 5:
        offset_weights = np.asarray([offset_weights[offset] for offset in offsets], dtype=np.float32)
    if len(offset_weights) != len(offsets):
        raise ValueError("--public-offset-weights must contain either 5 values or one value per selected offset")
    offset_weights = offset_weights / max(float(offset_weights.sum()), 1e-6)

    configs = [
        {
            "feature_set": feature_set,
            "norm_mode": norm_mode,
            "reducer": reducer,
            "model": model_name,
            "dep_weight": dep_weight,
        }
        for feature_set in feature_sets
        for norm_mode in norm_modes
        for reducer in reducers
        for model_name in model_names
        for dep_weight in dep_weights
    ]

    all_offset_rows = []
    all_outer_rows = []
    all_inner_top_rows = []
    all_full_rank_rows = []
    public_rank_parts = []
    weighted_proxy_scores: np.ndarray | None = None
    weighted_proxy_y: np.ndarray | None = None
    weighted_proxy_subjects: np.ndarray | None = None

    for offset_idx, offset in enumerate(offsets):
        print(f"\n=== offset {offset} ===", flush=True)
        offset_matrices: dict[tuple[str, str], np.ndarray] = {}
        public_matrices: dict[tuple[str, str], np.ndarray] = {}
        y_offset: np.ndarray | None = None
        subjects_offset: np.ndarray | None = None
        videos_offset: np.ndarray | None = None
        for feature_set in feature_sets:
            x_base = feature_data[feature_set].astype(np.float32)
            public_base = feature_data[f"public_{feature_set}"].astype(np.float32)
            for norm_mode in norm_modes:
                x_norm = normalize_by_subject(x_base, raw["subjects"], norm_mode)
                public_norm = normalize_by_subject(public_base, raw["public_subjects"], norm_mode)
                x_off, y_off, sub_off, video_off = offset_dataset(
                    x_norm,
                    raw["y"],
                    raw["subjects"],
                    raw["trial_ids"],
                    offset=offset,
                )
                if y_offset is None:
                    y_offset = y_off
                    subjects_offset = sub_off
                    videos_offset = video_off
                elif not np.array_equal(y_offset, y_off) or not np.array_equal(subjects_offset, sub_off):
                    raise ValueError("Offset dataset alignment mismatch")
                offset_matrices[(feature_set, norm_mode)] = x_off
                public_matrices[(feature_set, norm_mode)] = public_norm

        assert y_offset is not None and subjects_offset is not None and videos_offset is not None
        seed_oofs = []
        selected_counter: Counter[str] = Counter()
        for seed in seeds:
            oof, outer_df, inner_df = nested_offset_eval(
                offset_matrices,
                y_offset,
                subjects_offset,
                configs,
                seed=seed,
                folds=args.folds,
                inner_folds=args.inner_folds,
                k=args.topk_video,
            )
            seed_oofs.append(oof)
            selected_counter.update(outer_df["selected_config"].astype(str).tolist())
            metrics = score_with_groups(y_offset, oof, subjects_offset, args.topk_video)
            all_offset_rows.append({"offset": offset, "seed": seed, "variant": "nested_seed", **metrics})
            outer_df.insert(0, "offset", offset)
            inner_df.insert(0, "offset", offset)
            all_outer_rows.append(outer_df)
            all_inner_top_rows.append(inner_df)
            print(
                f"offset={offset} seed={seed} nested={metrics['accuracy']:.4f} "
                f"HC={metrics['hc_accuracy']:.4f} DEP={metrics['dep_accuracy']:.4f}",
                flush=True,
            )

        mean_oof = np.mean(np.vstack(seed_oofs), axis=0).astype(np.float32)
        mean_rank = rank_by_subject(mean_oof, subjects_offset)
        mean_metrics = score_with_groups(y_offset, mean_rank, subjects_offset, args.topk_video)
        all_offset_rows.append({"offset": offset, "seed": "mean", "variant": "nested_seed_mean_rank", **mean_metrics})
        if weighted_proxy_scores is None:
            weighted_proxy_scores = offset_weights[offset_idx] * mean_rank
            weighted_proxy_y = y_offset
            weighted_proxy_subjects = subjects_offset
        else:
            if not np.array_equal(weighted_proxy_y, y_offset) or not np.array_equal(weighted_proxy_subjects, subjects_offset):
                raise ValueError("Weighted offset alignment mismatch")
            weighted_proxy_scores += offset_weights[offset_idx] * mean_rank

        full_rank = full_cv_config_ranking(
            offset_matrices,
            y_offset,
            subjects_offset,
            configs,
            seed=seeds[0],
            folds=args.folds,
            k=args.topk_video,
        )
        full_rank.insert(0, "offset", offset)
        all_full_rank_rows.append(full_rank.head(50))
        frequent_name = selected_counter.most_common(1)[0][0]
        frequent_config = next(config for config in configs if config_name(config) == frequent_name)
        best_full_config = next(config for config in configs if config_name(config) == str(full_rank.iloc[0]["config"]))
        public_config = frequent_config
        public_scores_by_seed = []
        x_train = offset_matrices[(str(public_config["feature_set"]), str(public_config["norm_mode"]))]
        x_public = public_matrices[(str(public_config["feature_set"]), str(public_config["norm_mode"]))]
        for seed in seeds:
            public_scores_by_seed.append(
                fit_predict_config(
                    x_train,
                    y_offset,
                    subjects_offset,
                    x_public,
                    public_config,
                    seed=seed + 17000 + offset,
                )
            )
        public_rank_parts.append(rank_by_subject(np.mean(np.vstack(public_scores_by_seed), axis=0), raw["public_subjects"]))
        print(
            f"offset={offset} mean_rank={mean_metrics['accuracy']:.4f} "
            f"frequent_public={frequent_name} full_cv_best={config_name(best_full_config)} "
            f"freq={selected_counter[frequent_name]}",
            flush=True,
        )

    assert weighted_proxy_scores is not None and weighted_proxy_y is not None and weighted_proxy_subjects is not None
    weighted_proxy_scores = rank_by_subject(weighted_proxy_scores, weighted_proxy_subjects)
    weighted_metrics = score_with_groups(weighted_proxy_y, weighted_proxy_scores, weighted_proxy_subjects, args.topk_video)
    public_scores = np.sum(
        np.vstack([offset_weights[idx] * part for idx, part in enumerate(public_rank_parts)]),
        axis=0,
    ).astype(np.float32)
    public_scores = rank_by_subject(public_scores, raw["public_subjects"])
    public_labels = topk_by_subject(public_scores, raw["public_subjects"], args.topk_video)
    public_df = pd.DataFrame(
        {
            "user_id": raw["public_subjects"],
            "trial_id": raw["public_trial_ids"],
            "score": public_scores,
            "Emotion_label": public_labels,
        }
    )

    offset_df = pd.DataFrame(all_offset_rows)
    outer_df = pd.concat(all_outer_rows, ignore_index=True) if all_outer_rows else pd.DataFrame()
    inner_df = pd.concat(all_inner_top_rows, ignore_index=True) if all_inner_top_rows else pd.DataFrame()
    full_rank_df = pd.concat(all_full_rank_rows, ignore_index=True) if all_full_rank_rows else pd.DataFrame()
    weighted_df = pd.DataFrame([{"variant": "weighted_offsets", **weighted_metrics}])

    report_path = report_dir / f"ea_de_offset_{safe_tag}_report.xlsx"
    with pd.ExcelWriter(report_path) as writer:
        offset_df.to_excel(writer, sheet_name="offset_nested_metrics", index=False)
        weighted_df.to_excel(writer, sheet_name="weighted_proxy", index=False)
        outer_df.to_excel(writer, sheet_name="outer_folds", index=False)
        inner_df.to_excel(writer, sheet_name="inner_top_configs", index=False)
        full_rank_df.to_excel(writer, sheet_name="full_cv_config_rank", index=False)
        public_df.to_excel(writer, sheet_name="public_top4", index=False)

    submission_path = report_dir / f"public_test_submission_ea_de_offset_{safe_tag}_top4.xlsx"
    detail_path = report_dir / f"public_test_prediction_details_ea_de_offset_{safe_tag}_top4.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(detail_path, index=False)

    summary = {
        "weighted_offset_proxy_metrics": weighted_metrics,
        "offset_nested_metrics": offset_df.to_dict(orient="records"),
        "public_offset_weights": offset_weights.tolist(),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "feature_dims": {feature_set: int(feature_data[feature_set].shape[1]) for feature_set in feature_sets},
        "outputs": {
            "report": str(report_path),
            "submission": str(submission_path),
            "details": str(detail_path),
        },
        "args": vars(args),
    }
    (report_dir / f"ea_de_offset_{safe_tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / f"predictions_ea_de_offset_{safe_tag}.npz",
        public_probs=public_scores.astype(np.float32),
        public_subjects=raw["public_subjects"],
        public_trial_ids=raw["public_trial_ids"],
        public_labels=public_labels.astype(np.int64),
        weighted_proxy_scores=weighted_proxy_scores.astype(np.float32),
        weighted_proxy_y=weighted_proxy_y.astype(np.int64),
        weighted_proxy_subjects=weighted_proxy_subjects,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
