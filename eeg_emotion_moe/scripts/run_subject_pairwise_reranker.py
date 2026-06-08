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


DEFAULT_SOURCES = (
    "predictions_eegnet.npz,"
    "predictions_eegnet_mmd.npz,"
    "predictions_riemannian.npz,"
    "predictions_modma_softgate.npz,"
    "predictions_disease_aware_ensemble.npz,"
    "predictions_disease_pairwise.npz"
)
DEFAULT_WEIGHTS = "0.150,0.050,0.000,0.200,0.600,0.000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise reranker trained on final-ensemble OOF scores plus trial features."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sources", type=str, default=DEFAULT_SOURCES)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--norm-modes", type=str, default="center,zscore")
    parser.add_argument("--pair-features", type=str, default="base_only,base_rawdiff,base_rawdiff_abs")
    parser.add_argument("--models", type=str, default="logreg_c0.1,hgb,extra_trees")
    parser.add_argument("--max-pairs-per-subject", type=int, default=400)
    parser.add_argument("--topk-train", type=int, default=20)
    parser.add_argument("--topk-public", type=int, default=4)
    return parser.parse_args()


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn and joblib before running this script.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
    }


def topk_by_subject(scores: np.ndarray, subjects: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        pred[order[:k]] = 1
    return pred


def rank_by_subject(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    ranks = np.zeros(len(scores), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        if len(idx) == 1:
            ranks[idx] = 0.5
            continue
        order = np.argsort(scores[idx])
        local = np.empty(len(idx), dtype=np.float32)
        local[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        ranks[idx] = local
    return ranks


def topk_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> dict[str, float]:
    return binary_metrics(y, topk_by_subject(scores, subjects, k), scores)


def subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=str)
    hc = np.asarray([s for s in unique if str(s).upper().startswith("HC")], dtype=str)
    dep = np.asarray([s for s in unique if str(s).upper().startswith("DEP")], dtype=str)
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
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


def load_summary_features(output_dir: Path) -> dict[str, np.ndarray]:
    train = np.load(output_dir / "cache_contest_trial_summary.npz", allow_pickle=True)
    public = np.load(output_dir / "cache_public_trial_summary_from_raw.npz", allow_pickle=True)
    spans = [tuple(row) for row in train["spans"].tolist()]
    return {
        "x": train["x"].astype(np.float32),
        "y": train["y"].astype(np.int64),
        "subjects": train["subjects"].astype(str),
        "trial_ids": np.asarray([int(row[1]) for row in spans], dtype=np.int64),
        "public_x": public["x"].astype(np.float32),
        "public_subjects": public["subjects"].astype(str),
        "public_trial_ids": public["trial_ids"].astype(np.int64),
    }


def load_base_scores(output_dir: Path, sources: list[str], weights: np.ndarray):
    preds = [np.load(output_dir / source, allow_pickle=True) for source in sources]
    y = preds[0]["contest_y"].astype(np.int64)
    subjects = preds[0]["contest_subjects"].astype(str)
    public_subjects = preds[0]["public_subjects"].astype(str)
    public_trial_ids = preds[0]["public_trial_ids"].astype(np.int64)
    base_oof = sum(float(w) * pred["contest_oof"].astype(np.float32) for w, pred in zip(weights, preds))
    public = sum(float(w) * pred["public_probs"].astype(np.float32) for w, pred in zip(weights, preds))
    return {
        "y": y,
        "subjects": subjects,
        "public_subjects": public_subjects,
        "public_trial_ids": public_trial_ids,
        "base_oof": rank_by_subject(base_oof, subjects),
        "public_base": rank_by_subject(public, public_subjects),
    }


def pair_features(
    x_left: np.ndarray,
    x_right: np.ndarray,
    base_left: np.ndarray,
    base_right: np.ndarray,
    mode: str,
) -> np.ndarray:
    base_diff = (base_left - base_right).reshape(-1, 1)
    base_abs = np.abs(base_diff)
    if mode == "base_only":
        return np.hstack([base_diff, base_abs]).astype(np.float32)
    raw_diff = x_left - x_right
    if mode == "base_rawdiff":
        return np.hstack([base_diff, base_abs, raw_diff]).astype(np.float32)
    if mode == "base_rawdiff_abs":
        return np.hstack([base_diff, base_abs, raw_diff, np.abs(raw_diff)]).astype(np.float32)
    raise ValueError(mode)


def build_pairs(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    base: np.ndarray,
    *,
    mode: str,
    max_pairs_per_subject: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        pairs = np.asarray([(p, n) for p in pos for n in neg], dtype=np.int64)
        n_pairs = min(max_pairs_per_subject, len(pairs))
        chosen = rng.choice(len(pairs), size=n_pairs, replace=n_pairs > len(pairs))
        pos_idx = pairs[chosen, 0]
        neg_idx = pairs[chosen, 1]
        x_parts.append(pair_features(x[pos_idx], x[neg_idx], base[pos_idx], base[neg_idx], mode))
        y_parts.append(np.ones(n_pairs, dtype=np.int64))
        x_parts.append(pair_features(x[neg_idx], x[pos_idx], base[neg_idx], base[pos_idx], mode))
        y_parts.append(np.zeros(n_pairs, dtype=np.int64))
    return np.vstack(x_parts).astype(np.float32), np.concatenate(y_parts).astype(np.int64)


def make_model(model_name: str, seed: int):
    sk = require_sklearn()
    if model_name.startswith("logreg_c"):
        c_value = float(model_name.replace("logreg_c", ""))
        return sk["LogisticRegression"](C=c_value, solver="lbfgs", max_iter=3000, random_state=seed)
    if model_name == "hgb":
        return sk["HistGradientBoostingClassifier"](
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return sk["ExtraTreesClassifier"](
            n_estimators=220,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(model_name)


def fit_model(x_pair: np.ndarray, y_pair: np.ndarray, model_name: str, seed: int):
    sk = require_sklearn()
    scaler = sk["StandardScaler"]()
    x_scaled = scaler.fit_transform(x_pair)
    model = make_model(model_name, seed)
    model.fit(x_scaled, y_pair)
    return {"scaler": scaler, "model": model, "model_name": model_name}


def predict_pair_proba(bundle: dict[str, object], x_pair: np.ndarray) -> np.ndarray:
    scaler = bundle["scaler"]
    model = bundle["model"]
    x_scaled = scaler.transform(x_pair)  # type: ignore[union-attr]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_scaled)[:, 1].astype(np.float32)  # type: ignore[union-attr]
    raw = model.decision_function(x_scaled).astype(np.float32)  # type: ignore[union-attr]
    return (1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))).astype(np.float32)


def vote_scores(
    bundle: dict[str, object],
    x: np.ndarray,
    subjects: np.ndarray,
    base: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    out = np.zeros(len(x), dtype=np.float32)
    counts = np.zeros(len(x), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        left_positions: list[int] = []
        right_positions: list[int] = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                left_positions.append(idx[i])
                right_positions.append(idx[j])
        left = np.asarray(left_positions, dtype=np.int64)
        right = np.asarray(right_positions, dtype=np.int64)
        probs = predict_pair_proba(bundle, pair_features(x[left], x[right], base[left], base[right], mode))
        np.add.at(out, left, probs)
        np.add.at(out, right, 1.0 - probs)
        np.add.at(counts, left, 1.0)
        np.add.at(counts, right, 1.0)
    counts[counts == 0] = 1.0
    return (out / counts).astype(np.float32)


def subset_accuracy(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, prefix: str, k: int) -> float:
    mask = np.asarray([str(s).upper().startswith(prefix) for s in subjects])
    return float((topk_by_subject(scores[mask], subjects[mask], k) == y[mask]).mean())


def main() -> None:
    args = parse_args()
    require_sklearn()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    sources = [part.strip() for part in args.sources.split(",") if part.strip()]
    weights = np.asarray([float(part.strip()) for part in args.weights.split(",") if part.strip()], dtype=np.float32)
    weights = weights / weights.sum()
    features = load_summary_features(output_dir)
    base_data = load_base_scores(output_dir, sources, weights)
    x_raw = features["x"]
    y = base_data["y"]
    subjects = base_data["subjects"]
    base_oof = base_data["base_oof"]
    public_raw = features["public_x"]
    public_subjects = features["public_subjects"]
    public_base = base_data["public_base"]

    norm_modes = [part.strip() for part in args.norm_modes.split(",") if part.strip()]
    pair_modes = [part.strip() for part in args.pair_features.split(",") if part.strip()]
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    folds = subject_folds(subjects, args.folds, args.random_seed)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_oof: np.ndarray | None = None

    base_metrics = topk_metrics(y, base_oof, subjects, args.topk_train)
    print(f"base acc={base_metrics['accuracy']:.4f}", flush=True)
    for norm_mode in norm_modes:
        x_norm = normalize_by_subject(x_raw, subjects, norm_mode)
        for pair_mode in pair_modes:
            for model_name in model_names:
                name = f"{norm_mode}__{pair_mode}__{model_name}"
                oof = np.zeros(len(y), dtype=np.float32)
                print(f"\nconfig {name}", flush=True)
                for fold_idx, val_subjects in enumerate(folds, start=1):
                    val_mask = np.isin(subjects, val_subjects)
                    train_mask = ~val_mask
                    x_pair, y_pair = build_pairs(
                        x_norm[train_mask],
                        y[train_mask],
                        subjects[train_mask],
                        base_oof[train_mask],
                        mode=pair_mode,
                        max_pairs_per_subject=args.max_pairs_per_subject,
                        seed=args.random_seed + fold_idx * 97,
                    )
                    bundle = fit_model(x_pair, y_pair, model_name, args.random_seed + fold_idx * 101)
                    oof[val_mask] = vote_scores(
                        bundle,
                        x_norm[val_mask],
                        subjects[val_mask],
                        base_oof[val_mask],
                        mode=pair_mode,
                    )
                    fold_metrics = topk_metrics(y[val_mask], oof[val_mask], subjects[val_mask], args.topk_train)
                    print(f"  fold {fold_idx}: {fold_metrics['accuracy']:.4f}", flush=True)
                metrics = topk_metrics(y, oof, subjects, args.topk_train)
                row = {
                    "config": name,
                    "norm_mode": norm_mode,
                    "pair_mode": pair_mode,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "hc_accuracy": subset_accuracy(y, oof, subjects, "HC", args.topk_train),
                    "dep_accuracy": subset_accuracy(y, oof, subjects, "DEP", args.topk_train),
                    "corr_with_base": float(np.corrcoef(base_oof, oof)[0, 1]),
                }
                rows.append(row)
                print(f"  OOF={row['accuracy']:.4f} HC={row['hc_accuracy']:.4f} DEP={row['dep_accuracy']:.4f}", flush=True)
                if best is None or float(row["accuracy"]) > float(best["accuracy"]):
                    best = row
                    best_oof = oof.copy()
    assert best is not None and best_oof is not None
    result_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(result_df.head(20).to_string(index=False), flush=True)

    best_norm = str(best["norm_mode"])
    best_pair = str(best["pair_mode"])
    best_model = str(best["model"])
    x_full = normalize_by_subject(x_raw, subjects, best_norm)
    x_public = normalize_by_subject(public_raw, public_subjects, best_norm)
    x_pair, y_pair = build_pairs(
        x_full,
        y,
        subjects,
        base_oof,
        mode=best_pair,
        max_pairs_per_subject=args.max_pairs_per_subject,
        seed=args.random_seed + 9001,
    )
    final_bundle = fit_model(x_pair, y_pair, best_model, args.random_seed + 9007)
    public_scores = vote_scores(final_bundle, x_public, public_subjects, public_base, mode=best_pair)
    public_rank = rank_by_subject(public_scores, public_subjects)
    public_labels = topk_by_subject(public_scores, public_subjects, args.topk_public)
    public_df = pd.DataFrame(
        {
            "user_id": public_subjects,
            "trial_id": features["public_trial_ids"],
            "base_score": public_base,
            "rerank_score": public_scores,
            "rank_probability": public_rank,
            "Emotion_label": public_labels,
        }
    )
    report_dir = ensure_dir(output_dir / "subject_pairwise_reranker")
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(
        report_dir / "public_test_submission_subject_pairwise_reranker_top4.xlsx",
        index=False,
    )
    public_df.to_excel(report_dir / "public_test_prediction_details_subject_pairwise_reranker_top4.xlsx", index=False)
    with pd.ExcelWriter(report_dir / "subject_pairwise_reranker_report.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="configs", index=False)
        public_df.to_excel(writer, sheet_name="public_top4", index=False)

    np.savez_compressed(
        output_dir / "predictions_subject_pairwise_reranker.npz",
        contest_oof=rank_by_subject(best_oof, subjects),
        contest_raw_scores=best_oof.astype(np.float32),
        contest_y=y,
        contest_subjects=subjects,
        contest_trial_ids=features["trial_ids"],
        public_probs=public_rank,
        public_raw_scores=public_scores.astype(np.float32),
        public_subjects=public_subjects,
        public_trial_ids=features["public_trial_ids"],
        threshold=np.array([0.5], dtype=np.float32),
        best_config=np.array([json.dumps(best, ensure_ascii=False)]),
        base_oof=base_oof.astype(np.float32),
        public_base=public_base.astype(np.float32),
    )
    joblib.dump({"bundle": final_bundle, "best": best, "args": vars(args)}, report_dir / "subject_pairwise_reranker_model.joblib")
    report = {
        "base_top20_metrics": base_metrics,
        "best": best,
        "best_top20_metrics": topk_metrics(y, best_oof, subjects, args.topk_train),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "outputs": {
            "predictions": str(output_dir / "predictions_subject_pairwise_reranker.npz"),
            "report": str(report_dir / "subject_pairwise_reranker_report.xlsx"),
            "submission": str(report_dir / "public_test_submission_subject_pairwise_reranker_top4.xlsx"),
        },
    }
    (report_dir / "subject_pairwise_reranker_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
