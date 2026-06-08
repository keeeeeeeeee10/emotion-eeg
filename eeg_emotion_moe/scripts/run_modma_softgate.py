from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.modma_data import load_modma_summary_feature_matrix
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, ensure_dir, resolve_path


DEFAULT_MODMA_ROOT = PROJECT_ROOT / "MODMA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MODMA-informed MDD/HC gate and soft-routed emotion experts on contest training data."
    )
    parser.add_argument("--modma-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--refresh-modma-cache", action="store_true")
    parser.add_argument("--modma-max-segments-per-subject", type=int, default=30)
    parser.add_argument("--gate-modma-weight", type=float, default=0.25)
    parser.add_argument("--seed-weight", type=float, default=0.05)
    parser.add_argument("--soft-sweep-step", type=float, default=0.1)
    return parser.parse_args()


def require_sklearn():
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    return {
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "QuantileTransformer": QuantileTransformer,
        "StandardScaler": StandardScaler,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "clone": clone,
        "make_pipeline": make_pipeline,
    }


def load_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache: {path}\n"
            "Run scripts\\optimize_sklearn.py and scripts\\run_seed_pairwise_ranking.py once to build summary caches."
        )
    return np.load(path, allow_pickle=True)


def subject_type(subject: str) -> int:
    if str(subject).upper().startswith("DEP"):
        return 1
    if str(subject).upper().startswith("HC"):
        return 0
    raise ValueError(f"Cannot infer contest subject type from {subject}")


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int64)
    n = len(y)
    out = np.ones(n, dtype=np.float32)
    for cls in np.unique(y):
        mask = y == cls
        out[mask] = n / (len(np.unique(y)) * max(int(mask.sum()), 1))
    return out


def stratified_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=str)
    hc = unique[np.asarray([subject_type(s) == 0 for s in unique])]
    dep = unique[np.asarray([subject_type(s) == 1 for s in unique])]
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for idx, subject in enumerate(hc):
        folds[idx % n_folds].append(str(subject))
    for idx, subject in enumerate(dep):
        folds[idx % n_folds].append(str(subject))
    return [np.asarray(sorted(fold), dtype=str) for fold in folds]


def summarize_by_subject(x: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    ids: list[str] = []
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        block = x[idx]
        rows.append(
            np.concatenate(
                [
                    block.mean(axis=0),
                    block.std(axis=0),
                    np.median(block, axis=0),
                    np.percentile(block, 25, axis=0),
                    np.percentile(block, 75, axis=0),
                ]
            ).astype(np.float32)
        )
        ids.append(str(subject))
    return np.vstack(rows).astype(np.float32), np.asarray(ids, dtype=str)


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


def make_gate_models(seed: int):
    sk = require_sklearn()
    LogisticRegression = sk["LogisticRegression"]
    ExtraTreesClassifier = sk["ExtraTreesClassifier"]
    RandomForestClassifier = sk["RandomForestClassifier"]
    make_pipeline = sk["make_pipeline"]
    StandardScaler = sk["StandardScaler"]
    return {
        "logreg_l2_c0.2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, solver="lbfgs", max_iter=5000, random_state=seed),
        ),
        "logreg_l2_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed + 1),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 2,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 3,
        ),
    }


def make_expert_models(seed: int):
    sk = require_sklearn()
    LogisticRegression = sk["LogisticRegression"]
    ExtraTreesClassifier = sk["ExtraTreesClassifier"]
    RandomForestClassifier = sk["RandomForestClassifier"]
    HistGradientBoostingClassifier = sk["HistGradientBoostingClassifier"]
    make_pipeline = sk["make_pipeline"]
    StandardScaler = sk["StandardScaler"]
    return {
        "logreg_c0.2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, solver="lbfgs", max_iter=5000, random_state=seed),
        ),
        "logreg_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed + 1),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed + 2,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed + 3,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            random_state=seed + 4,
        ),
    }


def positive_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1].astype(np.float32)
    score = model.decision_function(x)
    score = np.asarray(score, dtype=np.float32)
    score = np.clip(score, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)


def fit_with_optional_weight(model, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    try:
        if sample_weight is None:
            model.fit(x, y)
        else:
            model.fit(x, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        if sample_weight is not None and hasattr(model, "steps"):
            final_name = model.steps[-1][0]
            try:
                model.fit(x, y, **{f"{final_name}__sample_weight": sample_weight})
                return model
            except (TypeError, ValueError):
                pass
        model.fit(x, y)
    return model


def train_gate_ensemble(
    x_contest_subject: np.ndarray,
    y_contest_subject: np.ndarray,
    x_modma_subject: np.ndarray,
    y_modma_subject: np.ndarray,
    *,
    modma_weight: float,
    seed: int,
):
    x = np.vstack([x_contest_subject, x_modma_subject]).astype(np.float32)
    y = np.concatenate([y_contest_subject, y_modma_subject]).astype(np.int64)
    weights = np.concatenate(
        [
            class_balanced_weights(y_contest_subject),
            class_balanced_weights(y_modma_subject) * float(modma_weight),
        ]
    ).astype(np.float32)
    models = {}
    for name, model in make_gate_models(seed).items():
        models[name] = fit_with_optional_weight(model, x, y, weights)
    return models


def predict_gate_ensemble(models: dict[str, object], x: np.ndarray) -> np.ndarray:
    probs = [positive_score(model, x) for model in models.values()]
    return np.mean(np.vstack(probs), axis=0).astype(np.float32)


def train_expert_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed_x: np.ndarray | None = None,
    seed_y: np.ndarray | None = None,
    seed_weight: float = 0.0,
    seed: int,
):
    x_fit = x.astype(np.float32)
    y_fit = y.astype(np.int64)
    weights = class_balanced_weights(y_fit)
    if seed_x is not None and seed_y is not None and seed_weight > 0:
        x_fit = np.vstack([x_fit, seed_x]).astype(np.float32)
        y_fit = np.concatenate([y_fit, seed_y.astype(np.int64)])
        weights = np.concatenate([weights, class_balanced_weights(seed_y) * float(seed_weight)]).astype(np.float32)
    models = {}
    for name, model in make_expert_models(seed).items():
        models[name] = fit_with_optional_weight(deepcopy(model), x_fit, y_fit, weights)
    return models


def predict_expert_ensemble(models: dict[str, object], x: np.ndarray) -> np.ndarray:
    probs = [positive_score(model, x) for model in models.values()]
    return np.mean(np.vstack(probs), axis=0).astype(np.float32)


def topk_by_subject(scores: np.ndarray, subjects: np.ndarray, k: int) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        labels[order[:k]] = 1
    return labels


def rank_probs_by_subject(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    probs = np.zeros(len(scores), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        local = scores[idx]
        if len(idx) == 1:
            probs[idx] = 0.5
            continue
        order = np.argsort(local)
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        probs[idx] = ranks
    return probs


def subject_topk_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> dict[str, float]:
    pred = topk_by_subject(scores, subjects, k)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def by_subject_accuracy(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> pd.DataFrame:
    pred = topk_by_subject(scores, subjects, k)
    rows = []
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        rows.append(
            {
                "subject": subject,
                "type": "DEP" if subject_type(subject) == 1 else "HC",
                "accuracy": float((pred[idx] == y[idx]).mean()),
                "positive_mean_score": float(scores[idx][y[idx] == 1].mean()),
                "neutral_mean_score": float(scores[idx][y[idx] == 0].mean()),
                "n": int(len(idx)),
            }
        )
    return pd.DataFrame(rows)


def weighted_sum(parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(next(iter(parts.values())), dtype=np.float32)
    total = 0.0
    for name, value in weights.items():
        if name not in parts:
            continue
        out += float(value) * parts[name]
        total += float(value)
    if total <= 0:
        raise ValueError("Weight sum must be positive")
    return (out / total).astype(np.float32)


def simplex3(step: float):
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-6:
        raise ValueError("--soft-sweep-step must divide 1.0")
    for a in range(units + 1):
        for b in range(units - a + 1):
            c = units - a - b
            yield a / units, b / units, c / units


def main() -> None:
    args = parse_args()
    require_sklearn()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    modma_root = resolve_path(args.modma_root, DEFAULT_MODMA_ROOT)

    contest_cache = load_required(output_dir / "cache_contest_trial_summary.npz")
    public_cache = load_required(output_dir / "cache_public_trial_summary_from_raw.npz")
    seed_cache = load_required(output_dir / "cache_seed_trial_summary.npz")

    x_contest = contest_cache["x"].astype(np.float32)
    y_contest = contest_cache["y"].astype(np.int64)
    contest_subjects = contest_cache["subjects"].astype(str)
    contest_trial_ids = np.asarray([row[1] for row in contest_cache["spans"].tolist()], dtype=np.int64)
    x_public = public_cache["x"].astype(np.float32)
    public_subjects = public_cache["subjects"].astype(str)
    public_trial_ids = public_cache["trial_ids"].astype(np.int64)
    x_seed = seed_cache["x"].astype(np.float32)
    y_seed = seed_cache["y"].astype(np.int64)

    print("loading/building MODMA 128-channel summary cache...", flush=True)
    x_modma, y_modma, modma_subjects, modma_meta = load_modma_summary_feature_matrix(
        modma_root,
        cache_path=output_dir / "cache_modma_128_trial_summary.npz",
        refresh_cache=args.refresh_modma_cache,
        segment_seconds=10.0,
        max_segments_per_subject=args.modma_max_segments_per_subject,
    )
    x_modma_subject, modma_subject_ids = summarize_by_subject(x_modma, modma_subjects)
    y_modma_subject = np.asarray(
        [int(y_modma[np.where(modma_subjects == sid)[0][0]]) for sid in modma_subject_ids],
        dtype=np.int64,
    )
    print(
        f"contest={x_contest.shape}, public={x_public.shape}, seed={x_seed.shape}, "
        f"modma_segments={x_modma.shape}, modma_subjects={len(modma_subject_ids)}",
        flush=True,
    )

    folds = stratified_subject_folds(contest_subjects, args.folds, args.random_seed)
    oof_parts = {
        "all": np.zeros(len(y_contest), dtype=np.float32),
        "hc": np.zeros(len(y_contest), dtype=np.float32),
        "dep": np.zeros(len(y_contest), dtype=np.float32),
        "soft": np.zeros(len(y_contest), dtype=np.float32),
        "gate": np.zeros(len(y_contest), dtype=np.float32),
        "center_soft": np.zeros(len(y_contest), dtype=np.float32),
        "center_all": np.zeros(len(y_contest), dtype=np.float32),
        "center_hc": np.zeros(len(y_contest), dtype=np.float32),
        "center_dep": np.zeros(len(y_contest), dtype=np.float32),
    }
    fold_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []

    for fold_idx, val_subjects in enumerate(folds, start=1):
        val_mask = np.isin(contest_subjects, val_subjects)
        train_mask = ~val_mask
        train_subjects = contest_subjects[train_mask]
        x_train = x_contest[train_mask]
        y_train = y_contest[train_mask]
        x_val = x_contest[val_mask]
        y_val = y_contest[val_mask]
        val_sub = contest_subjects[val_mask]
        train_types = np.asarray([subject_type(s) for s in train_subjects], dtype=np.int64)
        print(f"fold {fold_idx}/{args.folds}: train_subjects={len(set(train_subjects))} val_subjects={len(val_subjects)}", flush=True)

        x_train_subject, train_subject_ids = summarize_by_subject(x_train, train_subjects)
        y_train_subject = np.asarray([subject_type(sid) for sid in train_subject_ids], dtype=np.int64)
        x_val_subject, val_subject_ids = summarize_by_subject(x_val, val_sub)
        y_val_subject = np.asarray([subject_type(sid) for sid in val_subject_ids], dtype=np.int64)
        gate_models = train_gate_ensemble(
            x_train_subject,
            y_train_subject,
            x_modma_subject,
            y_modma_subject,
            modma_weight=args.gate_modma_weight,
            seed=args.random_seed + fold_idx * 101,
        )
        p_gate_subject = predict_gate_ensemble(gate_models, x_val_subject)
        gate_map = {sid: prob for sid, prob in zip(val_subject_ids.tolist(), p_gate_subject.tolist())}
        p_gate_trial = np.asarray([gate_map[s] for s in val_sub], dtype=np.float32)
        oof_parts["gate"][val_mask] = p_gate_trial
        gate_pred_subject = (p_gate_subject >= 0.5).astype(np.int64)
        gate_acc = float((gate_pred_subject == y_val_subject).mean())
        gate_rows.append(
            {
                "fold": fold_idx,
                "gate_subject_accuracy": gate_acc,
                "val_subjects": ",".join(val_subject_ids.tolist()),
                "p_dep_mean": float(p_gate_subject.mean()),
            }
        )
        print(f"  gate subject acc={gate_acc:.4f}", flush=True)

        hc_mask = train_types == 0
        dep_mask = train_types == 1
        all_models = train_expert_ensemble(
            x_train,
            y_train,
            seed_x=x_seed,
            seed_y=y_seed,
            seed_weight=args.seed_weight,
            seed=args.random_seed + fold_idx * 107,
        )
        hc_models = train_expert_ensemble(
            x_train[hc_mask],
            y_train[hc_mask],
            seed_x=x_seed,
            seed_y=y_seed,
            seed_weight=args.seed_weight,
            seed=args.random_seed + fold_idx * 109,
        )
        dep_models = train_expert_ensemble(
            x_train[dep_mask],
            y_train[dep_mask],
            seed_x=None,
            seed_y=None,
            seed_weight=0.0,
            seed=args.random_seed + fold_idx * 113,
        )
        p_all = predict_expert_ensemble(all_models, x_val)
        p_hc = predict_expert_ensemble(hc_models, x_val)
        p_dep = predict_expert_ensemble(dep_models, x_val)
        p_soft = (1.0 - p_gate_trial) * p_hc + p_gate_trial * p_dep
        centered_x_train = normalize_by_subject(x_train, train_subjects, "center")
        centered_x_val = normalize_by_subject(x_val, val_sub, "center")
        centered_seed = x_seed - x_seed.mean(axis=0, keepdims=True)
        centered_all_models = train_expert_ensemble(
            centered_x_train,
            y_train,
            seed_x=centered_seed,
            seed_y=y_seed,
            seed_weight=args.seed_weight,
            seed=args.random_seed + fold_idx * 127,
        )
        centered_hc_models = train_expert_ensemble(
            centered_x_train[hc_mask],
            y_train[hc_mask],
            seed_x=centered_seed,
            seed_y=y_seed,
            seed_weight=args.seed_weight,
            seed=args.random_seed + fold_idx * 131,
        )
        centered_dep_models = train_expert_ensemble(
            centered_x_train[dep_mask],
            y_train[dep_mask],
            seed=args.random_seed + fold_idx * 137,
        )
        p_center_all = predict_expert_ensemble(centered_all_models, centered_x_val)
        p_center_hc = predict_expert_ensemble(centered_hc_models, centered_x_val)
        p_center_dep = predict_expert_ensemble(centered_dep_models, centered_x_val)
        p_center_soft = (1.0 - p_gate_trial) * p_center_hc + p_gate_trial * p_center_dep

        fold_parts = {
            "all": p_all,
            "hc": p_hc,
            "dep": p_dep,
            "soft": p_soft,
            "center_all": p_center_all,
            "center_hc": p_center_hc,
            "center_dep": p_center_dep,
            "center_soft": p_center_soft,
        }
        for name, probs in fold_parts.items():
            oof_parts[name][val_mask] = probs.astype(np.float32)
            metrics = subject_topk_metrics(y_val, probs, val_sub, 20)
            fold_rows.append({"fold": fold_idx, "method": name, **metrics})
            print(f"  {name}: top20 acc={metrics['accuracy']:.4f}", flush=True)

    rows: list[dict[str, object]] = []
    for name, probs in oof_parts.items():
        if name == "gate":
            continue
        metrics = subject_topk_metrics(y_contest, probs, contest_subjects, 20)
        rows.append({"method": name, "weights": name, **metrics})

    for w_all, w_soft, w_center in simplex3(args.soft_sweep_step):
        if w_all == 0 and w_soft == 0 and w_center == 0:
            continue
        weights = {
            "all": w_all,
            "soft": w_soft,
            "center_soft": w_center,
        }
        scores = weighted_sum(oof_parts, weights)
        metrics = subject_topk_metrics(y_contest, scores, contest_subjects, 20)
        rows.append(
            {
                "method": "blend_all_soft_center",
                "weights": json.dumps(weights, sort_keys=True),
                **metrics,
            }
        )

    eval_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    best = eval_df.iloc[0].to_dict()
    best_weights = json.loads(str(best["weights"])) if str(best["weights"]).startswith("{") else {str(best["weights"]): 1.0}
    best_oof = weighted_sum(oof_parts, best_weights)
    print("\nOOF top20 summary:")
    print(eval_df.head(15).to_string(index=False), flush=True)
    print(f"\nbest={best}", flush=True)

    full_subject_x, full_subject_ids = summarize_by_subject(x_contest, contest_subjects)
    full_subject_y = np.asarray([subject_type(sid) for sid in full_subject_ids], dtype=np.int64)
    full_gate_models = train_gate_ensemble(
        full_subject_x,
        full_subject_y,
        x_modma_subject,
        y_modma_subject,
        modma_weight=args.gate_modma_weight,
        seed=args.random_seed + 9001,
    )
    public_subject_x, public_subject_ids = summarize_by_subject(x_public, public_subjects)
    public_gate_subject = predict_gate_ensemble(full_gate_models, public_subject_x)
    public_gate_map = {sid: prob for sid, prob in zip(public_subject_ids.tolist(), public_gate_subject.tolist())}
    public_gate_trial = np.asarray([public_gate_map[s] for s in public_subjects], dtype=np.float32)

    full_types = np.asarray([subject_type(s) for s in contest_subjects], dtype=np.int64)
    final_all = train_expert_ensemble(
        x_contest,
        y_contest,
        seed_x=x_seed,
        seed_y=y_seed,
        seed_weight=args.seed_weight,
        seed=args.random_seed + 9101,
    )
    final_hc = train_expert_ensemble(
        x_contest[full_types == 0],
        y_contest[full_types == 0],
        seed_x=x_seed,
        seed_y=y_seed,
        seed_weight=args.seed_weight,
        seed=args.random_seed + 9103,
    )
    final_dep = train_expert_ensemble(
        x_contest[full_types == 1],
        y_contest[full_types == 1],
        seed=args.random_seed + 9109,
    )
    public_all = predict_expert_ensemble(final_all, x_public)
    public_hc = predict_expert_ensemble(final_hc, x_public)
    public_dep = predict_expert_ensemble(final_dep, x_public)
    public_soft = (1.0 - public_gate_trial) * public_hc + public_gate_trial * public_dep

    centered_contest = normalize_by_subject(x_contest, contest_subjects, "center")
    centered_public = normalize_by_subject(x_public, public_subjects, "center")
    centered_seed_full = x_seed - x_seed.mean(axis=0, keepdims=True)
    final_center_all = train_expert_ensemble(
        centered_contest,
        y_contest,
        seed_x=centered_seed_full,
        seed_y=y_seed,
        seed_weight=args.seed_weight,
        seed=args.random_seed + 9127,
    )
    final_center_hc = train_expert_ensemble(
        centered_contest[full_types == 0],
        y_contest[full_types == 0],
        seed_x=centered_seed_full,
        seed_y=y_seed,
        seed_weight=args.seed_weight,
        seed=args.random_seed + 9131,
    )
    final_center_dep = train_expert_ensemble(
        centered_contest[full_types == 1],
        y_contest[full_types == 1],
        seed=args.random_seed + 9137,
    )
    public_center_all = predict_expert_ensemble(final_center_all, centered_public)
    public_center_hc = predict_expert_ensemble(final_center_hc, centered_public)
    public_center_dep = predict_expert_ensemble(final_center_dep, centered_public)
    public_center_soft = (1.0 - public_gate_trial) * public_center_hc + public_gate_trial * public_center_dep
    public_parts = {
        "all": public_all,
        "hc": public_hc,
        "dep": public_dep,
        "soft": public_soft,
        "center_all": public_center_all,
        "center_hc": public_center_hc,
        "center_dep": public_center_dep,
        "center_soft": public_center_soft,
    }
    public_scores = weighted_sum(public_parts, best_weights)
    public_rank_probs = rank_probs_by_subject(public_scores, public_subjects)
    public_pred = topk_by_subject(public_scores, public_subjects, 4)
    public_df = pd.DataFrame(
        {
            "user_id": public_subjects,
            "trial_id": public_trial_ids,
            "probability": public_scores,
            "rank_probability": public_rank_probs,
            "p_dep_gate": public_gate_trial,
            "score_all": public_all,
            "score_soft": public_soft,
            "score_center_soft": public_center_soft,
            "Emotion_label": public_pred,
        }
    )
    submission_path = output_dir / "public_test_submission_modma_softgate_top4.xlsx"
    detail_path = output_dir / "public_test_prediction_details_modma_softgate_top4.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(detail_path, index=False)

    np.savez_compressed(
        output_dir / "predictions_modma_softgate.npz",
        contest_oof=rank_probs_by_subject(best_oof, contest_subjects),
        contest_y=y_contest,
        contest_subjects=contest_subjects,
        contest_trial_ids=contest_trial_ids,
        public_probs=public_rank_probs,
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        threshold=np.array([0.5], dtype=np.float32),
        best_weights=np.array([json.dumps(best_weights, sort_keys=True)]),
        gate_oof=oof_parts["gate"],
        public_gate=public_gate_trial,
    )
    joblib.dump(
        {
            "gate_models": full_gate_models,
            "all_models": final_all,
            "hc_models": final_hc,
            "dep_models": final_dep,
            "center_all_models": final_center_all,
            "center_hc_models": final_center_hc,
            "center_dep_models": final_center_dep,
            "best_weights": best_weights,
            "args": vars(args),
            "modma_meta": modma_meta,
        },
        output_dir / "modma_softgate_model.joblib",
    )

    by_subject = by_subject_accuracy(y_contest, best_oof, contest_subjects, 20)
    with pd.ExcelWriter(output_dir / "modma_softgate_report.xlsx") as writer:
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="folds", index=False)
        pd.DataFrame(gate_rows).to_excel(writer, sheet_name="gate_folds", index=False)
        eval_df.to_excel(writer, sheet_name="oof_summary", index=False)
        by_subject.to_excel(writer, sheet_name="contest_by_subject", index=False)
        public_df.to_excel(writer, sheet_name="public_top4_details", index=False)

    report = {
        "best": best,
        "best_weights": best_weights,
        "best_oof_top20_metrics": subject_topk_metrics(y_contest, best_oof, contest_subjects, 20),
        "gate_subject_accuracy_mean": float(pd.DataFrame(gate_rows)["gate_subject_accuracy"].mean()),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "public_p_dep_by_user": {
            str(subject): float(public_df.loc[public_df["user_id"] == subject, "p_dep_gate"].mean())
            for subject in sorted(set(public_subjects.tolist()))
        },
        "outputs": {
            "submission": str(submission_path),
            "details": str(detail_path),
            "predictions": str(output_dir / "predictions_modma_softgate.npz"),
            "report": str(output_dir / "modma_softgate_report.xlsx"),
        },
        "args": vars(args),
        "modma_meta": {k: v for k, v in modma_meta.items() if k != "per_subject_segments"},
    }
    (output_dir / "modma_softgate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
