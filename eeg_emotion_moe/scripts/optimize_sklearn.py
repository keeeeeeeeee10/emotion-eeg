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

from seed_transfer.contest_data import (
    aggregate_trial_predictions,
    load_contest_training_trials,
    load_public_test_trials,
    trials_to_summary_feature_matrix,
)
from seed_transfer.model import best_threshold, binary_metrics
from seed_transfer.paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEED_ROOT, ensure_dir, resolve_path
from seed_transfer.seed_data import SeedLoadConfig, load_seed_raw_summary_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sklearn models on trial-summary EEG features.")
    parser.add_argument("--seed-root", type=str, default=None)
    parser.add_argument("--contest-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed-weight", type=float, default=0.08)
    parser.add_argument(
        "--models",
        type=str,
        default="logreg_c0.3,logreg_c1,extra_trees,hgb,ensemble",
        help="Comma-separated model names. Available: logreg_c0.3,logreg_c1,svc_rbf,extra_trees,random_forest,hgb,ensemble.",
    )
    parser.add_argument(
        "--source-modes",
        type=str,
        default="contest,seed_weighted",
        help="Comma-separated source modes: contest,seed_weighted.",
    )
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ImportError(
            "This script requires scikit-learn and joblib. Install with: python -m pip install scikit-learn joblib"
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int64)
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg)).astype(np.float32)


def stratified_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    hc = np.array([s for s in unique_subjects if str(s).startswith("HC")])
    dep = np.array([s for s in unique_subjects if str(s).startswith("DEP")])
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for idx, subject in enumerate(hc):
        folds[idx % n_folds].append(str(subject))
    for idx, subject in enumerate(dep):
        folds[idx % n_folds].append(str(subject))
    return [np.array(sorted(fold)) for fold in folds]


def load_seed_features(args: argparse.Namespace, seed_root: Path, output_dir: Path):
    cache_path = output_dir / "cache_seed_trial_summary.npz"
    if cache_path.exists() and not args.refresh_cache:
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        return data["x"].astype(np.float32), data["y"].astype(np.int64), meta
    config = SeedLoadConfig(
        seed_root=seed_root,
        source="raw_summary",
        raw_fs=200.0,
        window_seconds=args.window_seconds,
        segment_seconds=args.segment_seconds,
        random_seed=args.random_seed,
    )
    x, y, meta = load_seed_raw_summary_feature_matrix(config)
    np.savez_compressed(cache_path, x=x, y=y, meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return x, y, meta


def load_contest_features(args: argparse.Namespace, contest_root: Path, output_dir: Path):
    cache_path = output_dir / "cache_contest_trial_summary.npz"
    if cache_path.exists() and not args.refresh_cache:
        data = np.load(cache_path, allow_pickle=True)
        spans = [tuple(row) for row in data["spans"].tolist()]
        return (
            data["x"].astype(np.float32),
            data["y"].astype(np.int64),
            data["subjects"].astype(str),
            spans,
        )
    trials = load_contest_training_trials(contest_root)
    x, y, spans = trials_to_summary_feature_matrix(trials, window_seconds=args.window_seconds)
    if y is None:
        raise RuntimeError("Contest training labels missing")
    subjects = np.array([trial.user_id for trial in trials], dtype=str)
    np.savez_compressed(cache_path, x=x, y=y, subjects=subjects, spans=np.array(spans, dtype=object))
    return x, y, subjects, spans


def make_models(seed: int):
    sk = require_sklearn()
    LogisticRegression = sk["LogisticRegression"]
    SVC = sk["SVC"]
    ExtraTreesClassifier = sk["ExtraTreesClassifier"]
    RandomForestClassifier = sk["RandomForestClassifier"]
    HistGradientBoostingClassifier = sk["HistGradientBoostingClassifier"]
    return {
        "logreg_c0.3": LogisticRegression(C=0.3, solver="lbfgs", max_iter=4000, random_state=seed),
        "logreg_c1": LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, random_state=seed + 1),
        "svc_rbf": SVC(C=2.0, gamma="scale", probability=True, cache_size=800, random_state=seed + 2),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            n_jobs=-1,
            random_state=seed + 3,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            n_jobs=-1,
            random_state=seed + 4,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed + 5,
        ),
    }


ENSEMBLE_MEMBERS = ["logreg_c1", "extra_trees", "hgb"]


def positive_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 2:
            return proba[:, 1].astype(np.float32)
    score = model.decision_function(x)
    score = np.asarray(score, dtype=np.float32)
    score = np.clip(score, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)


def evaluate_scores(method: str, fold: int, y_train: np.ndarray, p_train: np.ndarray, y_test: np.ndarray, p_test: np.ndarray):
    threshold_row = best_threshold(y_train, p_train)
    threshold = float(threshold_row["threshold"])
    pred_test = (p_test >= threshold).astype(np.int64)
    metrics = binary_metrics(y_test, pred_test, p_test)
    oracle = best_threshold(y_test, p_test)
    return {
        "fold": fold,
        "method": method,
        "threshold": threshold,
        "train_threshold_accuracy": float(threshold_row["accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "tp": float(metrics["tp"]),
        "tn": float(metrics["tn"]),
        "fp": float(metrics["fp"]),
        "fn": float(metrics["fn"]),
        "oracle_threshold": float(oracle["threshold"]),
        "oracle_accuracy": float(oracle["accuracy"]),
    }


def fit_model(model, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    try:
        model.fit(x, y, sample_weight=weights)
    except TypeError:
        model.fit(x, y)
    return model


def prepare_training_data(
    source_mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_seed: np.ndarray,
    y_seed: np.ndarray,
    seed_weight: float,
):
    if source_mode == "contest":
        return x_train, y_train, class_balanced_weights(y_train)
    if source_mode == "seed_weighted":
        x = np.vstack([x_train, x_seed])
        y = np.concatenate([y_train, y_seed])
        w_contest = class_balanced_weights(y_train)
        w_seed = class_balanced_weights(y_seed) * seed_weight
        w = np.concatenate([w_contest, w_seed])
        return x, y, w
    raise ValueError(source_mode)


def train_model_family(
    model_name: str,
    source_mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_seed: np.ndarray,
    y_seed: np.ndarray,
    seed_weight: float,
    seed: int,
):
    sk = require_sklearn()
    scaler = sk["StandardScaler"]()
    x_fit, y_fit, w_fit = prepare_training_data(source_mode, x_train, y_train, x_seed, y_seed, seed_weight)
    scaler.fit(x_fit)
    x_fit_s = scaler.transform(x_fit)
    base_models = make_models(seed)
    if model_name == "ensemble":
        models = {}
        for member in ENSEMBLE_MEMBERS:
            models[member] = fit_model(deepcopy(base_models[member]), x_fit_s, y_fit, w_fit)
        return {"scaler": scaler, "models": models, "model_name": model_name, "source_mode": source_mode}
    model = fit_model(deepcopy(base_models[model_name]), x_fit_s, y_fit, w_fit)
    return {"scaler": scaler, "models": {model_name: model}, "model_name": model_name, "source_mode": source_mode}


def predict_family(bundle: dict, x: np.ndarray) -> np.ndarray:
    x_s = bundle["scaler"].transform(x)
    probs = [positive_score(model, x_s) for model in bundle["models"].values()]
    return np.mean(np.vstack(probs), axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    require_sklearn()
    seed_root = resolve_path(args.seed_root, DEFAULT_SEED_ROOT)
    contest_root = resolve_path(args.contest_root, DEFAULT_CONTEST_ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))

    print("loading cached/built trial-summary features...")
    x_seed, y_seed, seed_meta = load_seed_features(args, seed_root, output_dir)
    x_contest, y_contest, subjects, spans = load_contest_features(args, contest_root, output_dir)
    print(f"seed={x_seed.shape}, contest={x_contest.shape}, subjects={len(set(subjects.tolist()))}")

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    source_modes = [name.strip() for name in args.source_modes.split(",") if name.strip()]
    folds = stratified_subject_folds(subjects, args.folds, args.random_seed)
    rows: list[dict[str, object]] = []
    oof_store: dict[str, np.ndarray] = {}

    for fold_idx, test_subjects in enumerate(folds, start=1):
        test_mask = np.isin(subjects, test_subjects)
        train_mask = ~test_mask
        x_train, y_train = x_contest[train_mask], y_contest[train_mask]
        x_test, y_test = x_contest[test_mask], y_contest[test_mask]
        print(f"fold {fold_idx}/{args.folds}: train_subjects={len(set(subjects[train_mask]))} test_subjects={len(test_subjects)}", flush=True)
        for source_mode in source_modes:
            for model_name in model_names:
                method = f"{source_mode}__{model_name}"
                bundle = train_model_family(
                    model_name,
                    source_mode,
                    x_train,
                    y_train,
                    x_seed,
                    y_seed,
                    args.seed_weight,
                    args.random_seed + fold_idx * 100,
                )
                p_train = predict_family(bundle, x_train)
                p_test = predict_family(bundle, x_test)
                oof_store.setdefault(method, np.zeros(len(y_contest), dtype=np.float32))[test_mask] = p_test
                row = evaluate_scores(method, fold_idx, y_train, p_train, y_test, p_test)
                rows.append(row)
                print(f"  {method}: acc={row['accuracy']:.4f} oracle={row['oracle_accuracy']:.4f} thr={row['threshold']:.2f}", flush=True)

    cv_df = pd.DataFrame(rows)
    summary_df = (
        cv_df.groupby("method")[["accuracy", "balanced_accuracy", "oracle_accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary_df.columns
    ]
    summary_df = summary_df.sort_values("accuracy_mean", ascending=False)
    print("\nCV summary:")
    print(summary_df.to_string(index=False))

    report_path = output_dir / "sklearn_model_cv_report.xlsx"
    with pd.ExcelWriter(report_path) as writer:
        cv_df.to_excel(writer, sheet_name="folds", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    method_column = "method" if "method" in summary_df.columns else "method_"
    best_method = str(summary_df.iloc[0][method_column])
    best_source, best_model_name = best_method.split("__", 1)
    print(f"\nbest method: {best_method}")
    final_bundle = train_model_family(
        best_model_name,
        best_source,
        x_contest,
        y_contest,
        x_seed,
        y_seed,
        args.seed_weight,
        args.random_seed + 999,
    )
    p_all = predict_family(final_bundle, x_contest)
    threshold = float(best_threshold(y_contest, p_all)["threshold"])
    train_metrics = binary_metrics(y_contest, (p_all >= threshold).astype(np.int64), p_all)
    final_bundle["threshold"] = threshold
    final_bundle["feature_name"] = "trial_summary_bandpower"
    final_bundle["seed_meta"] = seed_meta
    final_bundle["args"] = vars(args)
    final_bundle["best_method"] = best_method
    model_path = output_dir / "sklearn_best_model.joblib"
    joblib.dump(final_bundle, model_path)

    public_trials = load_public_test_trials(contest_root)
    x_public, _, public_spans = trials_to_summary_feature_matrix(public_trials, window_seconds=args.window_seconds)
    p_public = predict_family(final_bundle, x_public)
    public_rows = aggregate_trial_predictions(public_spans, p_public, threshold=threshold)
    public_df = pd.DataFrame(public_rows)
    submission_path = output_dir / "public_test_submission_sklearn.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(output_dir / "public_test_prediction_details_sklearn.xlsx", index=False)
    np.savez_compressed(
        output_dir / "predictions_sklearn.npz",
        contest_oof=oof_store[best_method],
        contest_y=y_contest,
        contest_subjects=subjects,
        contest_trial_ids=np.asarray([span[1] for span in spans], dtype=np.int64),
        public_probs=p_public,
        public_subjects=np.asarray([span[0] for span in public_spans], dtype=str),
        public_trial_ids=np.asarray([span[1] for span in public_spans], dtype=np.int64),
        threshold=np.array([threshold], dtype=np.float32),
        best_method=np.array([best_method]),
    )

    json_report = {
        "best_method": best_method,
        "cv_report": str(report_path),
        "model_path": str(model_path),
        "submission_path": str(submission_path),
        "threshold": threshold,
        "train_metrics": train_metrics,
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "cv_top5": summary_df.head(5).to_dict(orient="records"),
        "args": vars(args),
    }
    json_path = output_dir / "sklearn_optimization_report.json"
    json_path.write_text(json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
