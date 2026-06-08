from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SEED-only pairwise ranker and validate by per-user top-k on contest training data."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-pairs-per-group", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--norm-modes", type=str, default="center,zscore,none")
    parser.add_argument("--c-values", type=str, default="0.03,0.1,0.3,1.0,3.0")
    return parser.parse_args()


def require_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    return LogisticRegression, SGDClassifier, StandardScaler


def load_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache: {path}\n"
            "Run scripts\\strict_seed_only.py once to build summary caches, or run the prior feature scripts."
        )
    return np.load(path, allow_pickle=True)


def normalize_by_group(x: np.ndarray, groups: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        block = x[idx]
        mean = block.mean(axis=0, keepdims=True)
        if mode == "center":
            out[idx] = block - mean
        elif mode == "zscore":
            std = block.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            out[idx] = (block - mean) / std
        else:
            raise ValueError(f"Unknown norm mode: {mode}")
    return out


def build_pairwise_dataset(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    max_pairs_per_group: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pairs = min(max_pairs_per_group, len(pos) * len(neg))
        pos_sample = rng.choice(pos, size=n_pairs, replace=True)
        neg_sample = rng.choice(neg, size=n_pairs, replace=True)
        diff_pos = x[pos_sample] - x[neg_sample]
        diff_neg = -diff_pos
        diffs.append(diff_pos.astype(np.float32))
        diffs.append(diff_neg.astype(np.float32))
        labels.append(np.ones(n_pairs, dtype=np.int64))
        labels.append(np.zeros(n_pairs, dtype=np.int64))
    if not diffs:
        raise RuntimeError("No pairwise samples built")
    return np.vstack(diffs).astype(np.float32), np.concatenate(labels).astype(np.int64)


def topk_by_group(scores: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        labels[order[:k]] = 1
    return labels


def rank_probs_by_group(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    probs = np.zeros(len(scores), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        order = idx[np.argsort(scores[idx])]
        if len(idx) == 1:
            probs[idx] = 0.5
            continue
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[np.argsort(scores[idx])] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        probs[idx] = ranks
    return probs


def group_accuracy(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        rows.append({"group": group, "accuracy": float((pred[idx] == y[idx]).mean()), "n": int(len(idx))})
    return pd.DataFrame(rows)


def score_model(model, scaler, x: np.ndarray) -> np.ndarray:
    xs = scaler.transform(x)
    return model.decision_function(xs).astype(np.float32)


def main() -> None:
    args = parse_args()
    LogisticRegression, SGDClassifier, StandardScaler = require_sklearn()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    seed_cache = load_required(output_dir / "cache_seed_strict_trial_summary.npz")
    contest_cache = load_required(output_dir / "cache_contest_trial_summary.npz")

    x_seed = seed_cache["x"].astype(np.float32)
    y_seed = seed_cache["y"].astype(np.int64)
    seed_groups = seed_cache["sessions"].astype(str)
    x_contest = contest_cache["x"].astype(np.float32)
    y_contest = contest_cache["y"].astype(np.int64)
    contest_groups = contest_cache["subjects"].astype(str)
    spans = [tuple(row) for row in contest_cache["spans"].tolist()]

    public_cache = load_required(output_dir / "cache_public_raw_trials_30x2500.npz")
    # Public summary features are not in the raw cache; use the existing sklearn detail-compatible cache if present.
    public_summary_path = output_dir / "cache_public_trial_summary_from_raw.npz"
    if public_summary_path.exists():
        public_summary = np.load(public_summary_path, allow_pickle=True)
        x_public = public_summary["x"].astype(np.float32)
        public_groups = public_summary["subjects"].astype(str)
        public_trial_ids = public_summary["trial_ids"].astype(np.int64)
    else:
        from seed_transfer.contest_data import load_public_test_trials, trials_to_summary_feature_matrix
        from seed_transfer.paths import DEFAULT_CONTEST_ROOT

        trials = load_public_test_trials(DEFAULT_CONTEST_ROOT)
        x_public, _, public_spans = trials_to_summary_feature_matrix(trials)
        public_groups = np.asarray([span[0] for span in public_spans], dtype=str)
        public_trial_ids = np.asarray([span[1] for span in public_spans], dtype=np.int64)
        np.savez_compressed(public_summary_path, x=x_public, subjects=public_groups, trial_ids=public_trial_ids)

    norm_modes = [part.strip() for part in args.norm_modes.split(",") if part.strip()]
    c_values = [float(part.strip()) for part in args.c_values.split(",") if part.strip()]
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_bundle = None

    for norm_mode in norm_modes:
        x_seed_norm = normalize_by_group(x_seed, seed_groups, norm_mode)
        x_contest_norm = normalize_by_group(x_contest, contest_groups, norm_mode)
        x_public_norm = normalize_by_group(x_public, public_groups, norm_mode)
        x_pair, y_pair = build_pairwise_dataset(
            x_seed_norm,
            y_seed,
            seed_groups,
            max_pairs_per_group=args.max_pairs_per_group,
            seed=args.random_seed,
        )
        scaler = StandardScaler()
        x_pair_scaled = scaler.fit_transform(x_pair)
        for c_value in c_values:
            alpha = 1.0 / max(c_value * len(y_pair), 1.0)
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=2000,
                tol=1e-4,
                fit_intercept=False,
                random_state=args.random_seed,
                n_jobs=-1,
            )
            model.fit(x_pair_scaled, y_pair)
            seed_scores = score_model(model, scaler, x_seed_norm)
            contest_scores = score_model(model, scaler, x_contest_norm)
            seed_pred = topk_by_group(seed_scores, seed_groups, 0)  # filled below per group half
            for group in sorted(set(seed_groups.tolist())):
                idx = np.where(seed_groups == group)[0]
                k = int((y_seed[idx] == 1).sum())
                order = idx[np.argsort(seed_scores[idx])[::-1]]
                seed_pred[idx] = 0
                seed_pred[order[:k]] = 1
            contest_pred = topk_by_group(contest_scores, contest_groups, 20)
            seed_acc = float((seed_pred == y_seed).mean())
            contest_acc = float((contest_pred == y_contest).mean())
            row = {
                "norm_mode": norm_mode,
                "C": c_value,
                "pair_samples": int(len(y_pair)),
                "seed_ranking_accuracy": seed_acc,
                "contest_validation_top20_accuracy": contest_acc,
            }
            rows.append(row)
            print(row, flush=True)
            if best is None or contest_acc > float(best["contest_validation_top20_accuracy"]):
                best = row
                best_bundle = (norm_mode, scaler, model, x_public_norm)

    assert best is not None and best_bundle is not None
    norm_mode, scaler, model, x_public_norm = best_bundle
    public_scores = score_model(model, scaler, x_public_norm)
    public_pred = topk_by_group(public_scores, public_groups, 4)
    contest_norm = normalize_by_group(x_contest, contest_groups, norm_mode)
    contest_scores = score_model(model, scaler, contest_norm)
    contest_rank_probs = rank_probs_by_group(contest_scores, contest_groups)
    public_rank_probs = rank_probs_by_group(public_scores, public_groups)
    submission = pd.DataFrame(
        {
            "user_id": public_groups,
            "trial_id": public_trial_ids,
            "score": public_scores,
            "rank_probability": public_rank_probs,
            "Emotion_label": public_pred,
        }
    )
    submission[["user_id", "trial_id", "Emotion_label"]].to_excel(
        output_dir / "public_test_submission_seed_pairwise_ranking_top4.xlsx",
        index=False,
    )
    submission.to_excel(output_dir / "public_test_prediction_details_seed_pairwise_ranking_top4.xlsx", index=False)
    joblib.dump(
        {"norm_mode": norm_mode, "scaler": scaler, "model": model, "best": best},
        output_dir / "seed_pairwise_ranker.joblib",
    )
    np.savez_compressed(
        output_dir / "predictions_seed_pairwise_ranking.npz",
        contest_oof=contest_rank_probs,
        contest_y=y_contest,
        contest_subjects=contest_groups,
        contest_trial_ids=np.asarray([span[1] for span in spans], dtype=np.int64),
        public_probs=public_rank_probs,
        public_subjects=public_groups,
        public_trial_ids=public_trial_ids,
        threshold=np.array([0.5], dtype=np.float32),
        best_norm_mode=np.array([norm_mode]),
        best_c=np.array([float(best["C"])], dtype=np.float32),
    )
    results = pd.DataFrame(rows).sort_values("contest_validation_top20_accuracy", ascending=False)
    by_user = group_accuracy(y_contest, topk_by_group(score_model(model, scaler, normalize_by_group(x_contest, contest_groups, norm_mode)), contest_groups, 20), contest_groups)
    with pd.ExcelWriter(output_dir / "seed_pairwise_ranking_report.xlsx") as writer:
        results.to_excel(writer, sheet_name="grid", index=False)
        by_user.to_excel(writer, sheet_name="contest_by_user", index=False)
        submission.to_excel(writer, sheet_name="public_top4_details", index=False)
    report = {
        "best": best,
        "public_positive_count": int((submission["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((submission["Emotion_label"] == 0).sum()),
        "outputs": {
            "submission": str(output_dir / "public_test_submission_seed_pairwise_ranking_top4.xlsx"),
            "report": str(output_dir / "seed_pairwise_ranking_report.xlsx"),
        },
    }
    (output_dir / "seed_pairwise_ranking_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
