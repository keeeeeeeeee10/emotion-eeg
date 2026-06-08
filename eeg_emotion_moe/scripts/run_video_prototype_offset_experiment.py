from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.model import binary_metrics
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stimulus/video prototype scoring with subject-wise offset/top4 validation."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--feature-cache", type=str, default="cache_ea_de_multifeature_ws1p0_2p0_4p0.npz")
    parser.add_argument("--outer-seeds", type=str, default="2026,2031,2042")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--offsets", type=str, default="1,3,4")
    parser.add_argument("--public-offset-weights", type=str, default="0.4,0.4,0.2")
    parser.add_argument("--feature-sets", type=str, default="summary,multi,ea_multi")
    parser.add_argument("--norm-modes", type=str, default="subject_center,subject_zscore")
    parser.add_argument("--reducers", type=str, default="select80,select160,select320,pca64")
    parser.add_argument("--metrics", type=str, default="cosine,euclidean")
    parser.add_argument("--aggregations", type=str, default="mean,max,logsum")
    parser.add_argument("--topk-video", type=int, default=4)
    return parser.parse_args()


def require_sklearn():
    try:
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn before running this script.") from exc
    return {
        "PCA": PCA,
        "SelectKBest": SelectKBest,
        "StandardScaler": StandardScaler,
        "f_classif": f_classif,
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
        local = np.empty(len(idx), dtype=np.float32)
        local[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        out[idx] = local
    return out


def score_with_groups(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> dict[str, float]:
    pred = topk_by_subject(scores, subjects, k)
    dep_mask = is_dep_subjects(subjects)
    metrics = binary_metrics(y, pred, scores)
    metrics["hc_accuracy"] = float((pred[~dep_mask] == y[~dep_mask]).mean())
    metrics["dep_accuracy"] = float((pred[dep_mask] == y[dep_mask]).mean())
    return metrics


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
            rows_videos.append(video_id)
    return (
        np.vstack(rows_x).astype(np.float32),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_subjects, dtype=str),
        np.asarray(rows_videos, dtype=np.int64),
    )


def fit_reducer(x_train: np.ndarray, y_train: np.ndarray, reducer: str, seed: int):
    sk = require_sklearn()
    scaler = sk["StandardScaler"]()
    x_train_s = scaler.fit_transform(x_train)
    transformer = None
    if reducer.startswith("select"):
        k = min(int(reducer.replace("select", "")), x_train.shape[1])
        transformer = sk["SelectKBest"](score_func=sk["f_classif"], k=k).fit(x_train_s, y_train)
        x_train_s = transformer.transform(x_train_s)
    elif reducer.startswith("pca"):
        n_components = min(int(reducer.replace("pca", "")), x_train.shape[0] - 1, x_train.shape[1])
        transformer = sk["PCA"](n_components=n_components, random_state=seed).fit(x_train_s)
        x_train_s = transformer.transform(x_train_s)
    elif reducer != "none":
        raise ValueError(f"Unknown reducer: {reducer}")
    return scaler, transformer, x_train_s.astype(np.float32)


def transform_reducer(scaler, transformer, x: np.ndarray) -> np.ndarray:
    out = scaler.transform(x)
    if transformer is not None:
        out = transformer.transform(out)
    return out.astype(np.float32)


def make_prototypes(x_train: np.ndarray, videos_train: np.ndarray) -> np.ndarray:
    prototypes = []
    for video_id in range(1, 9):
        idx = np.where(videos_train == video_id)[0]
        if len(idx) == 0:
            raise ValueError(f"No training samples for video {video_id}")
        prototypes.append(x_train[idx].mean(axis=0))
    return np.vstack(prototypes).astype(np.float32)


def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True) + 1e-8)).squeeze(axis)


def prototype_scores(
    x_eval: np.ndarray,
    prototypes: np.ndarray,
    *,
    metric: str,
    aggregation: str,
) -> np.ndarray:
    if metric == "cosine":
        x_norm = x_eval / (np.linalg.norm(x_eval, axis=1, keepdims=True) + 1e-8)
        p_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        sim = x_norm @ p_norm.T
    elif metric == "euclidean":
        dist = np.sum((x_eval[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        scale = np.median(dist)
        sim = -dist / max(float(scale), 1e-6)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    neg = sim[:, :4]
    pos = sim[:, 4:]
    if aggregation == "mean":
        return (pos.mean(axis=1) - neg.mean(axis=1)).astype(np.float32)
    if aggregation == "max":
        return (pos.max(axis=1) - neg.max(axis=1)).astype(np.float32)
    if aggregation == "logsum":
        return (logsumexp(pos, axis=1) - logsumexp(neg, axis=1)).astype(np.float32)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def evaluate_config(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    videos: np.ndarray,
    *,
    reducer: str,
    metric: str,
    aggregation: str,
    seed: int,
    folds: int,
    k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    oof = np.zeros(len(y), dtype=np.float32)
    rows = []
    for fold_idx, val_subjects in enumerate(subject_folds(subjects, folds, seed), start=1):
        val_mask = np.isin(subjects, val_subjects)
        train_mask = ~val_mask
        scaler, transformer, x_train = fit_reducer(x[train_mask], y[train_mask], reducer, seed + fold_idx)
        x_val = transform_reducer(scaler, transformer, x[val_mask])
        prototypes = make_prototypes(x_train, videos[train_mask])
        scores = prototype_scores(x_val, prototypes, metric=metric, aggregation=aggregation)
        oof[val_mask] = scores
        metrics = score_with_groups(y[val_mask], scores, subjects[val_mask], k)
        rows.append({"fold": fold_idx, **metrics})
    return oof, pd.DataFrame(rows)


def load_data(output_dir: Path, feature_cache: str) -> dict[str, np.ndarray]:
    summary = np.load(output_dir / "cache_contest_trial_summary.npz", allow_pickle=True)
    public_summary = np.load(output_dir / "cache_public_trial_summary_from_raw.npz", allow_pickle=True)
    spans = [tuple(row) for row in summary["spans"].tolist()]
    out = {
        "y": summary["y"].astype(np.int64),
        "subjects": summary["subjects"].astype(str),
        "trial_ids": np.asarray([int(row[1]) for row in spans], dtype=np.int64),
        "public_subjects": public_summary["subjects"].astype(str),
        "public_trial_ids": public_summary["trial_ids"].astype(np.int64),
        "summary": summary["x"].astype(np.float32),
        "public_summary": public_summary["x"].astype(np.float32),
    }
    cache_path = output_dir / feature_cache
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        for key in cache.files:
            out[key] = cache[key].astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    require_sklearn()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "main"
    report_dir = ensure_dir(output_dir / f"video_prototype_offset_{safe_tag}")
    data = load_data(output_dir, args.feature_cache)

    feature_sets = [part.strip() for part in args.feature_sets.split(",") if part.strip()]
    norm_modes = [part.strip() for part in args.norm_modes.split(",") if part.strip()]
    reducers = [part.strip() for part in args.reducers.split(",") if part.strip()]
    metrics = [part.strip() for part in args.metrics.split(",") if part.strip()]
    aggregations = [part.strip() for part in args.aggregations.split(",") if part.strip()]
    seeds = [int(part.strip()) for part in args.outer_seeds.split(",") if part.strip()]
    offsets = [int(part.strip()) for part in args.offsets.split(",") if part.strip()]
    offset_weights = np.asarray([float(part.strip()) for part in args.public_offset_weights.split(",") if part.strip()], dtype=np.float32)
    if len(offset_weights) == 5:
        offset_weights = np.asarray([offset_weights[offset] for offset in offsets], dtype=np.float32)
    if len(offset_weights) != len(offsets):
        raise ValueError("--public-offset-weights must contain either 5 values or one value per selected offset")
    offset_weights = offset_weights / max(float(offset_weights.sum()), 1e-6)

    all_config_rows = []
    all_fold_rows = []
    public_rank_parts = []
    weighted_scores: np.ndarray | None = None
    weighted_y: np.ndarray | None = None
    weighted_subjects: np.ndarray | None = None

    for offset_idx, offset in enumerate(offsets):
        print(f"\n=== offset {offset} ===", flush=True)
        best_row: dict[str, object] | None = None
        best_scores: np.ndarray | None = None
        best_payload: tuple[str, str, str, str, str] | None = None
        best_subjects: np.ndarray | None = None
        best_y: np.ndarray | None = None
        best_videos: np.ndarray | None = None
        for feature_set in feature_sets:
            if feature_set not in data or f"public_{feature_set}" not in data:
                print(f"skip missing feature_set={feature_set}", flush=True)
                continue
            for norm_mode in norm_modes:
                x_norm = normalize_by_subject(data[feature_set], data["subjects"], norm_mode)
                x_off, y_off, sub_off, video_off = offset_dataset(
                    x_norm,
                    data["y"],
                    data["subjects"],
                    data["trial_ids"],
                    offset=offset,
                )
                for reducer in reducers:
                    for metric in metrics:
                        for aggregation in aggregations:
                            seed_scores = []
                            seed_metrics = []
                            fold_parts = []
                            for seed in seeds:
                                oof, fold_df = evaluate_config(
                                    x_off,
                                    y_off,
                                    sub_off,
                                    video_off,
                                    reducer=reducer,
                                    metric=metric,
                                    aggregation=aggregation,
                                    seed=seed,
                                    folds=args.folds,
                                    k=args.topk_video,
                                )
                                ranked = rank_by_subject(oof, sub_off)
                                seed_scores.append(ranked)
                                sm = score_with_groups(y_off, ranked, sub_off, args.topk_video)
                                seed_metrics.append(sm)
                                fold_df.insert(0, "seed", seed)
                                fold_parts.append(fold_df)
                            mean_scores = np.mean(np.vstack(seed_scores), axis=0).astype(np.float32)
                            mean_scores = rank_by_subject(mean_scores, sub_off)
                            mean_metrics = score_with_groups(y_off, mean_scores, sub_off, args.topk_video)
                            row = {
                                "offset": offset,
                                "feature_set": feature_set,
                                "norm_mode": norm_mode,
                                "reducer": reducer,
                                "metric": metric,
                                "aggregation": aggregation,
                                **mean_metrics,
                                "seed_accuracy_min": float(min(m["accuracy"] for m in seed_metrics)),
                                "seed_dep_min": float(min(m["dep_accuracy"] for m in seed_metrics)),
                            }
                            all_config_rows.append(row)
                            fold_df_all = pd.concat(fold_parts, ignore_index=True)
                            fold_df_all.insert(0, "aggregation", aggregation)
                            fold_df_all.insert(0, "metric", metric)
                            fold_df_all.insert(0, "reducer", reducer)
                            fold_df_all.insert(0, "norm_mode", norm_mode)
                            fold_df_all.insert(0, "feature_set", feature_set)
                            fold_df_all.insert(0, "offset", offset)
                            all_fold_rows.append(fold_df_all)
                            key = (
                                row["accuracy"],
                                row["seed_accuracy_min"],
                                row["dep_accuracy"],
                                row["hc_accuracy"],
                            )
                            old_key = None
                            if best_row is not None:
                                old_key = (
                                    best_row["accuracy"],
                                    best_row["seed_accuracy_min"],
                                    best_row["dep_accuracy"],
                                    best_row["hc_accuracy"],
                                )
                            if old_key is None or key > old_key:
                                best_row = row
                                best_scores = mean_scores
                                best_payload = (feature_set, norm_mode, reducer, metric, aggregation)
                                best_subjects = sub_off
                                best_y = y_off
                                best_videos = video_off
        assert best_row is not None and best_scores is not None and best_payload is not None
        assert best_subjects is not None and best_y is not None and best_videos is not None
        print(
            f"offset={offset} best={best_row['accuracy']:.4f} HC={best_row['hc_accuracy']:.4f} "
            f"DEP={best_row['dep_accuracy']:.4f} {best_payload}",
            flush=True,
        )
        if weighted_scores is None:
            weighted_scores = offset_weights[offset_idx] * best_scores
            weighted_y = best_y
            weighted_subjects = best_subjects
        else:
            if not np.array_equal(weighted_y, best_y) or not np.array_equal(weighted_subjects, best_subjects):
                raise ValueError("Weighted offset alignment mismatch")
            weighted_scores += offset_weights[offset_idx] * best_scores

        feature_set, norm_mode, reducer, metric, aggregation = best_payload
        x_train_full = normalize_by_subject(data[feature_set], data["subjects"], norm_mode)
        x_public_full = normalize_by_subject(data[f"public_{feature_set}"], data["public_subjects"], norm_mode)
        x_off, y_off, sub_off, video_off = offset_dataset(
            x_train_full,
            data["y"],
            data["subjects"],
            data["trial_ids"],
            offset=offset,
        )
        public_seed_scores = []
        for seed in seeds:
            scaler, transformer, x_train = fit_reducer(x_off, y_off, reducer, seed + offset * 100)
            prototypes = make_prototypes(x_train, video_off)
            x_public = transform_reducer(scaler, transformer, x_public_full)
            public_seed_scores.append(
                rank_by_subject(
                    prototype_scores(x_public, prototypes, metric=metric, aggregation=aggregation),
                    data["public_subjects"],
                )
            )
        public_rank_parts.append(np.mean(np.vstack(public_seed_scores), axis=0).astype(np.float32))

    assert weighted_scores is not None and weighted_y is not None and weighted_subjects is not None
    weighted_scores = rank_by_subject(weighted_scores, weighted_subjects)
    weighted_metrics = score_with_groups(weighted_y, weighted_scores, weighted_subjects, args.topk_video)
    public_scores = np.sum(
        np.vstack([offset_weights[idx] * rank_by_subject(part, data["public_subjects"]) for idx, part in enumerate(public_rank_parts)]),
        axis=0,
    ).astype(np.float32)
    public_scores = rank_by_subject(public_scores, data["public_subjects"])
    public_labels = topk_by_subject(public_scores, data["public_subjects"], args.topk_video)
    public_df = pd.DataFrame(
        {
            "user_id": data["public_subjects"],
            "trial_id": data["public_trial_ids"],
            "score": public_scores,
            "Emotion_label": public_labels,
        }
    )

    config_df = pd.DataFrame(all_config_rows).sort_values(["offset", "accuracy", "seed_accuracy_min", "dep_accuracy"], ascending=[True, False, False, False])
    fold_df = pd.concat(all_fold_rows, ignore_index=True) if all_fold_rows else pd.DataFrame()
    weighted_df = pd.DataFrame([{"variant": "weighted_offsets", **weighted_metrics}])
    report_path = report_dir / f"video_prototype_offset_{safe_tag}_report.xlsx"
    with pd.ExcelWriter(report_path) as writer:
        config_df.to_excel(writer, sheet_name="config_metrics", index=False)
        config_df.groupby("offset").head(10).to_excel(writer, sheet_name="top_by_offset", index=False)
        weighted_df.to_excel(writer, sheet_name="weighted_proxy", index=False)
        fold_df.to_excel(writer, sheet_name="fold_metrics", index=False)
        public_df.to_excel(writer, sheet_name="public_top4", index=False)

    submission_path = report_dir / f"public_test_submission_video_prototype_offset_{safe_tag}_top4.xlsx"
    detail_path = report_dir / f"public_test_prediction_details_video_prototype_offset_{safe_tag}_top4.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(detail_path, index=False)
    summary = {
        "weighted_offset_proxy_metrics": weighted_metrics,
        "best_by_offset": config_df.groupby("offset").head(1).to_dict(orient="records"),
        "public_offset_weights": offset_weights.tolist(),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "outputs": {
            "report": str(report_path),
            "submission": str(submission_path),
            "details": str(detail_path),
        },
        "args": vars(args),
    }
    (report_dir / f"video_prototype_offset_{safe_tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / f"predictions_video_prototype_offset_{safe_tag}.npz",
        public_probs=public_scores.astype(np.float32),
        public_subjects=data["public_subjects"],
        public_trial_ids=data["public_trial_ids"],
        public_labels=public_labels.astype(np.int64),
        weighted_proxy_scores=weighted_scores.astype(np.float32),
        weighted_proxy_y=weighted_y.astype(np.int64),
        weighted_proxy_subjects=weighted_subjects,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
