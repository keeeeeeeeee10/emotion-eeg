from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.model import binary_metrics
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


DEFAULT_SOURCES = (
    "predictions_disease_aware_ensemble.npz,"
    "predictions_supcon_mlp_main_h96.npz,"
    "predictions_eegnet_coral.npz,"
    "predictions_modma_softgate.npz,"
    "predictions_subject_pairwise_reranker.npz,"
    "predictions_video_prototype_offset_main_proto.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subject-wise validation for public-like weighted-offset source ensembles. "
            "Regular prediction files are projected to likely training offsets; prototype files "
            "may provide precomputed weighted_proxy_scores."
        )
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--sources", type=str, default=DEFAULT_SOURCES)
    parser.add_argument("--offsets", type=str, default="1,3,4")
    parser.add_argument("--offset-weights", type=str, default="0.4,0.4,0.2")
    parser.add_argument("--outer-seeds", type=str, default="2026,2031,2042")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--combo-max-size", type=int, default=4)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--topk-video", type=int, default=4)
    return parser.parse_args()


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


def simplex_weights(n: int, step: float):
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-6:
        raise ValueError("--weight-step must divide 1.0")

    def rec(remaining: int, slots: int):
        if slots == 1:
            yield [remaining]
            return
        for value in range(remaining + 1):
            for tail in rec(remaining - value, slots - 1):
                yield [value] + tail

    for counts in rec(units, n):
        if any(count == 0 for count in counts):
            continue
        yield np.asarray(counts, dtype=np.float32) / units


def config_name(indices: tuple[int, ...], weights: np.ndarray, names: list[str]) -> str:
    parts = [f"{names[idx]}:{weights[pos]:.2f}" for pos, idx in enumerate(indices)]
    return "+".join(parts)


def weighted_scores(matrix: np.ndarray, indices: tuple[int, ...], weights: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    scores = np.sum(matrix[:, indices] * weights.reshape(1, -1), axis=1).astype(np.float32)
    return rank_by_subject(scores, subjects)


def generate_configs(n_sources: int, max_size: int, step: float):
    import itertools

    for size in range(1, max_size + 1):
        for indices in itertools.combinations(range(n_sources), size):
            for weights in simplex_weights(size, step):
                yield indices, weights


def load_regular_source(path: Path, offsets: list[int], offset_weights: np.ndarray) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    y = data["contest_y"].astype(np.int64)
    subjects = data["contest_subjects"].astype(str)
    trial_ids = data["contest_trial_ids"].astype(np.int64)
    scores = data["contest_oof"].astype(np.float32)
    video_ids, offs = video_and_offset(trial_ids)
    total = None
    proxy_y = None
    proxy_subjects = None
    for local_idx, offset in enumerate(offsets):
        rows = []
        rows_y = []
        rows_subjects = []
        for subject in sorted(set(subjects.tolist())):
            for video_id in range(1, 9):
                idx = np.where((subjects == subject) & (video_ids == video_id) & (offs == offset))[0]
                if len(idx) != 1:
                    raise ValueError(f"{path.name}: expected one row for {subject} video {video_id} offset {offset}, got {len(idx)}")
                rows.append(scores[idx[0]])
                rows_y.append(int(y[idx[0]]))
                rows_subjects.append(str(subject))
        rows_arr = np.asarray(rows, dtype=np.float32)
        y_arr = np.asarray(rows_y, dtype=np.int64)
        subjects_arr = np.asarray(rows_subjects, dtype=str)
        ranked = rank_by_subject(rows_arr, subjects_arr)
        if total is None:
            total = float(offset_weights[local_idx]) * ranked
            proxy_y = y_arr
            proxy_subjects = subjects_arr
        else:
            if not np.array_equal(proxy_y, y_arr) or not np.array_equal(proxy_subjects, subjects_arr):
                raise ValueError(f"{path.name}: proxy row alignment mismatch")
            total += float(offset_weights[local_idx]) * ranked
    assert total is not None and proxy_y is not None and proxy_subjects is not None
    public_subjects = data["public_subjects"].astype(str)
    return {
        "proxy_scores": rank_by_subject(total, proxy_subjects),
        "proxy_y": proxy_y,
        "proxy_subjects": proxy_subjects,
        "public_scores": rank_by_subject(data["public_probs"].astype(np.float32), public_subjects),
        "public_subjects": public_subjects,
        "public_trial_ids": data["public_trial_ids"].astype(np.int64),
    }


def load_proxy_source(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    public_subjects = data["public_subjects"].astype(str)
    return {
        "proxy_scores": rank_by_subject(data["weighted_proxy_scores"].astype(np.float32), data["weighted_proxy_subjects"].astype(str)),
        "proxy_y": data["weighted_proxy_y"].astype(np.int64),
        "proxy_subjects": data["weighted_proxy_subjects"].astype(str),
        "public_scores": rank_by_subject(data["public_probs"].astype(np.float32), public_subjects),
        "public_subjects": public_subjects,
        "public_trial_ids": data["public_trial_ids"].astype(np.int64),
    }


def load_source(path: Path, offsets: list[int], offset_weights: np.ndarray) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    files = set(data.files)
    data.close()
    if {"weighted_proxy_scores", "weighted_proxy_y", "weighted_proxy_subjects"}.issubset(files):
        return load_proxy_source(path)
    return load_regular_source(path, offsets, offset_weights)


def select_best_config(
    matrix: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    configs: list[tuple[tuple[int, ...], np.ndarray]],
    names: list[str],
    *,
    k: int,
) -> tuple[tuple[int, ...], np.ndarray, dict[str, object]]:
    best_key = None
    best_payload = None
    for indices, weights in configs:
        scores = weighted_scores(matrix[train_mask], indices, weights, subjects[train_mask])
        metrics = score_with_groups(y[train_mask], scores, subjects[train_mask], k)
        key = (metrics["accuracy"], metrics["dep_accuracy"], metrics["hc_accuracy"], -len(indices))
        if best_key is None or key > best_key:
            best_key = key
            best_payload = (
                indices,
                weights,
                {
                    "selected_config": config_name(indices, weights, names),
                    "selected_indices": ",".join(str(idx) for idx in indices),
                    "selected_weights": ",".join(f"{w:.3f}" for w in weights),
                    "train_accuracy": metrics["accuracy"],
                    "train_hc_accuracy": metrics["hc_accuracy"],
                    "train_dep_accuracy": metrics["dep_accuracy"],
                },
            )
    assert best_payload is not None
    return best_payload


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "main"
    report_dir = ensure_dir(output_dir / f"weighted_offset_source_ensemble_{safe_tag}")
    source_names = [part.strip() for part in args.sources.split(",") if part.strip()]
    offsets = [int(part.strip()) for part in args.offsets.split(",") if part.strip()]
    offset_weights = np.asarray([float(part.strip()) for part in args.offset_weights.split(",") if part.strip()], dtype=np.float32)
    if len(offset_weights) == 5:
        offset_weights = np.asarray([offset_weights[offset] for offset in offsets], dtype=np.float32)
    if len(offset_weights) != len(offsets):
        raise ValueError("--offset-weights must contain either 5 values or one value per selected offset")
    offset_weights = offset_weights / max(float(offset_weights.sum()), 1e-6)
    seeds = [int(part.strip()) for part in args.outer_seeds.split(",") if part.strip()]

    loaded = [load_source(output_dir / name, offsets, offset_weights) for name in source_names]
    y = loaded[0]["proxy_y"]
    subjects = loaded[0]["proxy_subjects"]
    public_subjects = loaded[0]["public_subjects"]
    public_trial_ids = loaded[0]["public_trial_ids"]
    for name, item in zip(source_names[1:], loaded[1:]):
        if not np.array_equal(y, item["proxy_y"]) or not np.array_equal(subjects, item["proxy_subjects"]):
            raise ValueError(f"Proxy alignment mismatch: {name}")
        if not np.array_equal(public_subjects, item["public_subjects"]) or not np.array_equal(public_trial_ids, item["public_trial_ids"]):
            raise ValueError(f"Public alignment mismatch: {name}")
    matrix = np.vstack([item["proxy_scores"] for item in loaded]).T.astype(np.float32)
    public_matrix = np.vstack([item["public_scores"] for item in loaded]).T.astype(np.float32)
    configs = list(generate_configs(len(source_names), args.combo_max_size, args.weight_step))

    source_rows = []
    for idx, name in enumerate(source_names):
        metrics = score_with_groups(y, matrix[:, idx], subjects, args.topk_video)
        source_rows.append({"source": name, **metrics})
        print(f"source {name}: acc={metrics['accuracy']:.4f} HC={metrics['hc_accuracy']:.4f} DEP={metrics['dep_accuracy']:.4f}", flush=True)

    seed_rows = []
    fold_rows = []
    seed_oofs = []
    selected_counter: Counter[str] = Counter()
    for seed in seeds:
        oof = np.zeros(len(y), dtype=np.float32)
        for fold_idx, val_subjects in enumerate(subject_folds(subjects, args.folds, seed), start=1):
            val_mask = np.isin(subjects, val_subjects)
            train_mask = ~val_mask
            indices, weights, selected = select_best_config(
                matrix,
                y,
                subjects,
                train_mask,
                configs,
                source_names,
                k=args.topk_video,
            )
            selected_counter.update([str(selected["selected_config"])])
            val_scores = weighted_scores(matrix[val_mask], indices, weights, subjects[val_mask])
            oof[val_mask] = val_scores
            metrics = score_with_groups(y[val_mask], val_scores, subjects[val_mask], args.topk_video)
            fold_rows.append({"seed": seed, "fold": fold_idx, **selected, **metrics})
            print(
                f"seed={seed} fold={fold_idx} val={metrics['accuracy']:.4f} "
                f"HC={metrics['hc_accuracy']:.4f} DEP={metrics['dep_accuracy']:.4f} {selected['selected_config']}",
                flush=True,
            )
        seed_oofs.append(oof)
        seed_metrics = score_with_groups(y, oof, subjects, args.topk_video)
        seed_rows.append({"seed": seed, **seed_metrics})
        print(
            f"seed={seed} nested acc={seed_metrics['accuracy']:.4f} "
            f"HC={seed_metrics['hc_accuracy']:.4f} DEP={seed_metrics['dep_accuracy']:.4f}",
            flush=True,
        )

    mean_oof = rank_by_subject(np.mean(np.vstack(seed_oofs), axis=0), subjects)
    nested_mean_metrics = score_with_groups(y, mean_oof, subjects, args.topk_video)
    frequent_config_name = selected_counter.most_common(1)[0][0]
    frequent_indices = None
    frequent_weights = None
    for indices, weights in configs:
        if config_name(indices, weights, source_names) == frequent_config_name:
            frequent_indices = indices
            frequent_weights = weights
            break
    assert frequent_indices is not None and frequent_weights is not None

    full_indices, full_weights, full_selected = select_best_config(
        matrix,
        y,
        subjects,
        np.ones(len(y), dtype=bool),
        configs,
        source_names,
        k=args.topk_video,
    )
    full_scores = weighted_scores(matrix, full_indices, full_weights, subjects)
    full_metrics = score_with_groups(y, full_scores, subjects, args.topk_video)
    frequent_scores = weighted_scores(matrix, frequent_indices, frequent_weights, subjects)
    frequent_metrics = score_with_groups(y, frequent_scores, subjects, args.topk_video)

    public_scores = weighted_scores(public_matrix, frequent_indices, frequent_weights, public_subjects)
    public_labels = topk_by_subject(public_scores, public_subjects, args.topk_video)
    public_df = pd.DataFrame(
        {
            "user_id": public_subjects,
            "trial_id": public_trial_ids,
            "score": public_scores,
            "Emotion_label": public_labels,
        }
    )

    report_path = report_dir / f"weighted_offset_source_ensemble_{safe_tag}_report.xlsx"
    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame(source_rows).to_excel(writer, sheet_name="source_metrics", index=False)
        pd.DataFrame(seed_rows).to_excel(writer, sheet_name="nested_seed_metrics", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="outer_folds", index=False)
        pd.DataFrame([{"variant": "nested_seed_mean_rank", **nested_mean_metrics}]).to_excel(writer, sheet_name="nested_mean", index=False)
        pd.DataFrame([{"variant": "frequent_config_full_proxy", "config": frequent_config_name, **frequent_metrics}]).to_excel(writer, sheet_name="frequent_config", index=False)
        pd.DataFrame([{"variant": "full_proxy_best", **full_selected, **full_metrics}]).to_excel(writer, sheet_name="full_proxy_best", index=False)
        pd.DataFrame(selected_counter.most_common(), columns=["config", "count"]).to_excel(writer, sheet_name="selected_frequency", index=False)
        public_df.to_excel(writer, sheet_name="public_top4", index=False)

    submission_path = report_dir / f"public_test_submission_weighted_offset_source_ensemble_{safe_tag}_top4.xlsx"
    detail_path = report_dir / f"public_test_prediction_details_weighted_offset_source_ensemble_{safe_tag}_top4.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(detail_path, index=False)
    summary = {
        "source_metrics": source_rows,
        "nested_seed_metrics": seed_rows,
        "nested_seed_mean_rank": nested_mean_metrics,
        "frequent_config": frequent_config_name,
        "frequent_config_full_proxy_metrics": frequent_metrics,
        "full_proxy_best": {**full_selected, **full_metrics},
        "selected_frequency_top10": selected_counter.most_common(10),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "outputs": {
            "report": str(report_path),
            "submission": str(submission_path),
            "details": str(detail_path),
        },
        "args": vars(args),
    }
    (report_dir / f"weighted_offset_source_ensemble_{safe_tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / f"predictions_weighted_offset_source_ensemble_{safe_tag}.npz",
        proxy_oof=mean_oof.astype(np.float32),
        proxy_y=y.astype(np.int64),
        proxy_subjects=subjects,
        public_probs=public_scores.astype(np.float32),
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        public_labels=public_labels.astype(np.int64),
        source_names=np.asarray(source_names, dtype=object),
        frequent_config=np.asarray([frequent_config_name], dtype=object),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
