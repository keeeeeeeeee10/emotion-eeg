from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.model import binary_metrics
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train subject-level pairwise rankers with a DEP-focused expert and soft public routing."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--norm-modes", type=str, default="center,zscore")
    parser.add_argument("--c-values", type=str, default="0.03,0.1,0.3,1.0")
    parser.add_argument("--max-pairs-per-subject", type=int, default=400)
    parser.add_argument("--dep-repeat", type=int, default=3)
    parser.add_argument("--gate-source", type=str, default="predictions_modma_softgate.npz")
    parser.add_argument("--gate-alphas", type=str, default="0,0.25,0.5,0.75,1.0")
    return parser.parse_args()


def require_sklearn():
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    return SGDClassifier, StandardScaler


def load_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cache/artifact: {path}")
    return np.load(path, allow_pickle=True)


def is_dep(subjects: np.ndarray) -> np.ndarray:
    return np.asarray([str(s).upper().startswith("DEP") for s in subjects], dtype=bool)


def stratified_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=str)
    dep = unique[is_dep(unique)]
    hc = unique[~is_dep(unique)]
    rng = np.random.default_rng(seed)
    rng.shuffle(dep)
    rng.shuffle(hc)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for idx, subject in enumerate(hc):
        folds[idx % n_folds].append(str(subject))
    for idx, subject in enumerate(dep):
        folds[idx % n_folds].append(str(subject))
    return [np.asarray(sorted(fold), dtype=str) for fold in folds]


def normalize_by_subject(x: np.ndarray, subjects: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        block = x[idx]
        mean = block.mean(axis=0, keepdims=True)
        if mode == "center":
            out[idx] = block - mean
        elif mode == "zscore":
            std = block.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            out[idx] = (block - mean) / std
        else:
            raise ValueError(mode)
    return out


def build_pairwise_dataset(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    subject_mask: np.ndarray | None,
    max_pairs_per_subject: int,
    dep_repeat: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    selected_subjects = sorted(set(subjects.tolist()))
    for subject in selected_subjects:
        idx = np.where(subjects == subject)[0]
        if subject_mask is not None and not bool(subject_mask[idx[0]]):
            continue
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_total = len(pos) * len(neg)
        n_pairs = min(max_pairs_per_subject, n_total)
        repeat = dep_repeat if str(subject).upper().startswith("DEP") else 1
        pair_indices = np.asarray([(pi, ni) for pi in pos for ni in neg], dtype=np.int64)
        for _ in range(repeat):
            selected = rng.choice(len(pair_indices), size=n_pairs, replace=n_pairs > n_total)
            pos_sample = pair_indices[selected, 0]
            neg_sample = pair_indices[selected, 1]
            diff_pos = x[pos_sample] - x[neg_sample]
            diff_neg = -diff_pos
            diffs.append(diff_pos.astype(np.float32))
            diffs.append(diff_neg.astype(np.float32))
            labels.append(np.ones(n_pairs, dtype=np.int64))
            labels.append(np.zeros(n_pairs, dtype=np.int64))
    if not diffs:
        raise RuntimeError("No pairwise samples built")
    return np.vstack(diffs).astype(np.float32), np.concatenate(labels).astype(np.int64)


def fit_ranker(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    seed: int,
):
    SGDClassifier, StandardScaler = require_sklearn()
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    alpha = 1.0 / max(c_value * len(y), 1.0)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=3000,
        tol=1e-4,
        fit_intercept=False,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(xs, y)
    return scaler, model


def score_ranker(bundle, x: np.ndarray) -> np.ndarray:
    scaler, model = bundle
    return model.decision_function(scaler.transform(x)).astype(np.float32)


def topk_by_subject(scores: np.ndarray, subjects: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        pred[idx[np.argsort(scores[idx])[::-1][:k]]] = 1
    return pred


def rank_probs_by_subject(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        local = scores[idx]
        if len(idx) == 1:
            out[idx] = 0.5
            continue
        order = np.argsort(local)
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        out[idx] = ranks
    return out


def topk_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int = 20) -> dict[str, float]:
    pred = topk_by_subject(scores, subjects, k)
    return binary_metrics(y, pred, scores)


def subset_acc(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, mask: np.ndarray) -> float:
    pred = topk_by_subject(scores[mask], subjects[mask], 20)
    return float((pred == y[mask]).mean())


def by_subject_accuracy(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    pred = topk_by_subject(scores, subjects, 20)
    rows = []
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        rows.append(
            {
                "subject": subject,
                "type": "DEP" if str(subject).upper().startswith("DEP") else "HC",
                "accuracy": float((pred[idx] == y[idx]).mean()),
                "positive_mean_score": float(scores[idx][y[idx] == 1].mean()),
                "neutral_mean_score": float(scores[idx][y[idx] == 0].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    require_sklearn()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    contest_cache = load_required(output_dir / "cache_contest_trial_summary.npz")
    public_cache = load_required(output_dir / "cache_public_trial_summary_from_raw.npz")
    gate = load_required(output_dir / args.gate_source)

    x = contest_cache["x"].astype(np.float32)
    y = contest_cache["y"].astype(np.int64)
    subjects = contest_cache["subjects"].astype(str)
    trial_ids = np.asarray([row[1] for row in contest_cache["spans"].tolist()], dtype=np.int64)
    x_public = public_cache["x"].astype(np.float32)
    public_subjects = public_cache["subjects"].astype(str)
    public_trial_ids = public_cache["trial_ids"].astype(np.int64)
    p_dep_oof = gate["gate_oof"].astype(np.float32)
    p_dep_public = gate["public_gate"].astype(np.float32)
    dep_mask_all = is_dep(subjects)
    hc_mask_all = ~dep_mask_all
    dep_prior = float(dep_mask_all.mean())

    norm_modes = [part.strip() for part in args.norm_modes.split(",") if part.strip()]
    c_values = [float(part.strip()) for part in args.c_values.split(",") if part.strip()]
    gate_alphas = [float(part.strip()) for part in args.gate_alphas.split(",") if part.strip()]
    folds = stratified_subject_folds(subjects, args.folds, args.random_seed)

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_oof: np.ndarray | None = None
    best_public_scores: np.ndarray | None = None
    best_bundles = None
    best_normed_full = None
    best_normed_public = None

    for norm_mode in norm_modes:
        x_norm = normalize_by_subject(x, subjects, norm_mode)
        x_public_norm = normalize_by_subject(x_public, public_subjects, norm_mode)
        for c_value in c_values:
            oof_all = np.zeros(len(y), dtype=np.float32)
            oof_hc = np.zeros(len(y), dtype=np.float32)
            oof_dep = np.zeros(len(y), dtype=np.float32)
            for fold_idx, val_subjects in enumerate(folds, start=1):
                val_mask = np.isin(subjects, val_subjects)
                train_mask = ~val_mask
                x_train, y_train, sub_train = x_norm[train_mask], y[train_mask], subjects[train_mask]
                x_val = x_norm[val_mask]
                train_dep = is_dep(sub_train)
                pair_all = build_pairwise_dataset(
                    x_train,
                    y_train,
                    sub_train,
                    subject_mask=None,
                    max_pairs_per_subject=args.max_pairs_per_subject,
                    dep_repeat=args.dep_repeat,
                    seed=args.random_seed + fold_idx * 101,
                )
                pair_hc = build_pairwise_dataset(
                    x_train,
                    y_train,
                    sub_train,
                    subject_mask=~train_dep,
                    max_pairs_per_subject=args.max_pairs_per_subject,
                    dep_repeat=1,
                    seed=args.random_seed + fold_idx * 103,
                )
                pair_dep = build_pairwise_dataset(
                    x_train,
                    y_train,
                    sub_train,
                    subject_mask=train_dep,
                    max_pairs_per_subject=args.max_pairs_per_subject,
                    dep_repeat=args.dep_repeat,
                    seed=args.random_seed + fold_idx * 107,
                )
                model_all = fit_ranker(*pair_all, c_value=c_value, seed=args.random_seed + fold_idx * 109)
                model_hc = fit_ranker(*pair_hc, c_value=c_value, seed=args.random_seed + fold_idx * 113)
                model_dep = fit_ranker(*pair_dep, c_value=c_value, seed=args.random_seed + fold_idx * 127)
                oof_all[val_mask] = score_ranker(model_all, x_val)
                oof_hc[val_mask] = score_ranker(model_hc, x_val)
                oof_dep[val_mask] = score_ranker(model_dep, x_val)

            for alpha in gate_alphas:
                effective_gate = (1.0 - alpha) * dep_prior + alpha * p_dep_oof
                scores_soft = ((1.0 - effective_gate) * oof_hc + effective_gate * oof_dep).astype(np.float32)
                scores_oracle = np.where(dep_mask_all, oof_dep, oof_hc).astype(np.float32)
                candidates = {
                    "all": oof_all,
                    "soft_hc_dep": scores_soft,
                    "oracle_type_hc_dep": scores_oracle,
                    "blend_all_soft": (0.5 * oof_all + 0.5 * scores_soft).astype(np.float32),
                }
                for method, scores in candidates.items():
                    metrics = topk_metrics(y, scores, subjects)
                    row = {
                        "norm_mode": norm_mode,
                        "C": c_value,
                        "gate_alpha": alpha,
                        "method": method,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "hc_accuracy": subset_acc(y, scores, subjects, hc_mask_all),
                        "dep_accuracy": subset_acc(y, scores, subjects, dep_mask_all),
                    }
                    rows.append(row)
                    if method != "oracle_type_hc_dep" and (best is None or row["accuracy"] > float(best["accuracy"])):
                        best = row
                        best_oof = scores.copy()
                        best_normed_full = x_norm
                        best_normed_public = x_public_norm

    assert best is not None and best_oof is not None and best_normed_full is not None and best_normed_public is not None
    print(pd.DataFrame(rows).sort_values("accuracy", ascending=False).head(20).to_string(index=False), flush=True)

    best_norm_mode = str(best["norm_mode"])
    best_c = float(best["C"])
    x_full = normalize_by_subject(x, subjects, best_norm_mode)
    x_pub = normalize_by_subject(x_public, public_subjects, best_norm_mode)
    pair_all_full = build_pairwise_dataset(
        x_full,
        y,
        subjects,
        subject_mask=None,
        max_pairs_per_subject=args.max_pairs_per_subject,
        dep_repeat=args.dep_repeat,
        seed=args.random_seed + 9001,
    )
    pair_hc_full = build_pairwise_dataset(
        x_full,
        y,
        subjects,
        subject_mask=hc_mask_all,
        max_pairs_per_subject=args.max_pairs_per_subject,
        dep_repeat=1,
        seed=args.random_seed + 9003,
    )
    pair_dep_full = build_pairwise_dataset(
        x_full,
        y,
        subjects,
        subject_mask=dep_mask_all,
        max_pairs_per_subject=args.max_pairs_per_subject,
        dep_repeat=args.dep_repeat,
        seed=args.random_seed + 9007,
    )
    final_all = fit_ranker(*pair_all_full, c_value=best_c, seed=args.random_seed + 9011)
    final_hc = fit_ranker(*pair_hc_full, c_value=best_c, seed=args.random_seed + 9013)
    final_dep = fit_ranker(*pair_dep_full, c_value=best_c, seed=args.random_seed + 9017)
    public_all = score_ranker(final_all, x_pub)
    public_hc = score_ranker(final_hc, x_pub)
    public_dep = score_ranker(final_dep, x_pub)
    alpha = float(best["gate_alpha"])
    public_effective_gate = (1.0 - alpha) * dep_prior + alpha * p_dep_public
    if str(best["method"]) == "all":
        public_scores = public_all
    elif str(best["method"]) == "soft_hc_dep":
        public_scores = ((1.0 - public_effective_gate) * public_hc + public_effective_gate * public_dep).astype(np.float32)
    elif str(best["method"]) == "blend_all_soft":
        public_soft = ((1.0 - public_effective_gate) * public_hc + public_effective_gate * public_dep).astype(np.float32)
        public_scores = (0.5 * public_all + 0.5 * public_soft).astype(np.float32)
    else:
        raise ValueError(f"Unsupported public method: {best['method']}")

    public_rank = rank_probs_by_subject(public_scores, public_subjects)
    public_pred = topk_by_subject(public_scores, public_subjects, 4)
    public_df = pd.DataFrame(
        {
            "user_id": public_subjects,
            "trial_id": public_trial_ids,
            "score": public_scores,
            "rank_probability": public_rank,
            "p_dep_gate": p_dep_public,
            "p_dep_effective": public_effective_gate,
            "Emotion_label": public_pred,
        }
    )
    submission_path = output_dir / "public_test_submission_disease_pairwise_top4.xlsx"
    detail_path = output_dir / "public_test_prediction_details_disease_pairwise_top4.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(detail_path, index=False)
    contest_rank = rank_probs_by_subject(best_oof, subjects)
    np.savez_compressed(
        output_dir / "predictions_disease_pairwise.npz",
        contest_oof=contest_rank,
        contest_y=y,
        contest_subjects=subjects,
        contest_trial_ids=trial_ids,
        public_probs=public_rank,
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        threshold=np.array([0.5], dtype=np.float32),
        best_norm_mode=np.array([best_norm_mode]),
        best_c=np.array([best_c], dtype=np.float32),
        best_method=np.array([str(best["method"])]),
        best_gate_alpha=np.array([alpha], dtype=np.float32),
    )
    joblib.dump(
        {
            "all": final_all,
            "hc": final_hc,
            "dep": final_dep,
            "best": best,
            "args": vars(args),
        },
        output_dir / "disease_pairwise_rankers.joblib",
    )
    result_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    with pd.ExcelWriter(output_dir / "disease_pairwise_ranking_report.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="grid", index=False)
        by_subject_accuracy(y, best_oof, subjects).to_excel(writer, sheet_name="contest_by_subject", index=False)
        public_df.to_excel(writer, sheet_name="public_top4_details", index=False)
    report = {
        "best": best,
        "best_top20_metrics": topk_metrics(y, best_oof, subjects),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "outputs": {
            "submission": str(submission_path),
            "details": str(detail_path),
            "predictions": str(output_dir / "predictions_disease_pairwise.npz"),
            "report": str(output_dir / "disease_pairwise_ranking_report.xlsx"),
        },
    }
    (output_dir / "disease_pairwise_ranking_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
