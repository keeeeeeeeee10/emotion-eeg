from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.channels import seed_to_contest_mapping
from seed_transfer.contest_data import load_contest_training_trials, trials_to_summary_feature_matrix
from seed_transfer.features import bandpower_de_features, summarize_window_features
from seed_transfer.model import best_threshold, binary_metrics
from seed_transfer.paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEED_ROOT, ensure_dir, resolve_path
from seed_transfer.seed_data import SEED_LABELS, list_seed_raw_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict experiment: train/test only on SEED, validate only on contest training data."
    )
    parser.add_argument("--seed-root", type=str, default=None)
    parser.add_argument("--contest-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--models", type=str, default="logreg_c0.3,logreg_c1,hgb,ensemble")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def require_sklearn():
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, LogisticRegression, HistGradientBoostingClassifier, ExtraTreesClassifier


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg)).astype(np.float32)


def seed_trial_variables(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, shape, _ in whosmat(path):
        match = re.search(r"_eeg(\d+)$", name)
        if match and len(shape) == 2 and shape[0] == 62:
            out[int(match.group(1))] = name
    if len(out) < 15:
        raise RuntimeError(f"Expected 15 trial variables in {path}, found {len(out)}")
    return out


def build_seed_strict_features(args: argparse.Namespace, seed_root: Path, output_dir: Path):
    cache_path = output_dir / "cache_seed_strict_trial_summary.npz"
    if cache_path.exists() and not args.refresh_cache:
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["x"].astype(np.float32),
            data["y"].astype(np.int64),
            data["subjects"].astype(str),
            data["sessions"].astype(str),
            data["meta"].item(),
        )

    mapping = seed_to_contest_mapping()
    files = list_seed_raw_files(seed_root, "Preprocessed_EEG")
    windows_per_segment = int(round(args.segment_seconds / args.window_seconds))
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    subjects: list[str] = []
    sessions: list[str] = []

    for path in files:
        subject = path.stem.split("_", 1)[0]
        trial_vars = seed_trial_variables(path)
        selected = [(idx, int(label)) for idx, label in enumerate(SEED_LABELS, start=1) if int(label) in (0, 1)]
        mat = loadmat(path, variable_names=[trial_vars[idx] for idx, _ in selected])
        for trial_idx, label in selected:
            arr = np.asarray(mat[trial_vars[trial_idx]], dtype=np.float32)[mapping.seed_indices, :]
            window_features = bandpower_de_features(arr, fs=200.0, window_seconds=args.window_seconds)
            n_segments = window_features.shape[0] // windows_per_segment
            for segment_idx in range(n_segments):
                start = segment_idx * windows_per_segment
                stop = start + windows_per_segment
                x_parts.append(summarize_window_features(window_features[start:stop]))
                y_parts.append(label)
                subjects.append(subject)
                sessions.append(path.stem)

    x = np.vstack(x_parts).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    subjects_arr = np.asarray(subjects, dtype=str)
    sessions_arr = np.asarray(sessions, dtype=str)
    meta = json.dumps(
        {
            "source": "SEED Preprocessed_EEG only",
            "samples": int(len(y)),
            "features": int(x.shape[1]),
            "subjects": int(len(set(subjects))),
            "positive": int((y == 1).sum()),
            "neutral": int((y == 0).sum()),
        },
        ensure_ascii=False,
    )
    np.savez_compressed(cache_path, x=x, y=y, subjects=subjects_arr, sessions=sessions_arr, meta=np.array(meta))
    return x, y, subjects_arr, sessions_arr, meta


def build_contest_validation_features(args: argparse.Namespace, contest_root: Path, output_dir: Path):
    cache_path = output_dir / "cache_contest_strict_validation_summary.npz"
    if cache_path.exists() and not args.refresh_cache:
        data = np.load(cache_path, allow_pickle=True)
        spans = [tuple(row) for row in data["spans"].tolist()]
        return data["x"].astype(np.float32), data["y"].astype(np.int64), spans
    trials = load_contest_training_trials(contest_root)
    x, y, spans = trials_to_summary_feature_matrix(trials, window_seconds=args.window_seconds)
    if y is None:
        raise RuntimeError("Contest validation labels missing")
    np.savez_compressed(cache_path, x=x, y=y, spans=np.array(spans, dtype=object))
    return x, y, spans


def make_models(seed: int):
    _, LogisticRegression, HistGradientBoostingClassifier, ExtraTreesClassifier = require_sklearn()
    return {
        "logreg_c0.3": LogisticRegression(C=0.3, solver="lbfgs", max_iter=5000, random_state=seed),
        "logreg_c1": LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed + 1),
        "hgb": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed + 2,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed + 3,
        ),
    }


def fit_one(model_name: str, x: np.ndarray, y: np.ndarray, seed: int):
    StandardScaler, _, _, _ = require_sklearn()
    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)
    models = make_models(seed)
    if model_name == "ensemble":
        fitted = {}
        for member in ["logreg_c1", "hgb", "extra_trees"]:
            model = deepcopy(models[member])
            try:
                model.fit(x_s, y, sample_weight=class_balanced_weights(y))
            except TypeError:
                model.fit(x_s, y)
            fitted[member] = model
        return {"scaler": scaler, "models": fitted, "model_name": model_name}
    model = deepcopy(models[model_name])
    try:
        model.fit(x_s, y, sample_weight=class_balanced_weights(y))
    except TypeError:
        model.fit(x_s, y)
    return {"scaler": scaler, "models": {model_name: model}, "model_name": model_name}


def predict(bundle: dict, x: np.ndarray) -> np.ndarray:
    x_s = bundle["scaler"].transform(x)
    probs = []
    for model in bundle["models"].values():
        if hasattr(model, "predict_proba"):
            probs.append(model.predict_proba(x_s)[:, 1].astype(np.float32))
        else:
            pred = model.predict(x_s).astype(np.float32)
            probs.append(pred)
    return np.mean(np.vstack(probs), axis=0).astype(np.float32)


def subject_folds(subjects: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    return [np.array(sorted(unique_subjects[i::folds])) for i in range(folds)]


def score_with_threshold(y_true: np.ndarray, prob: np.ndarray, threshold: float):
    pred = (prob >= threshold).astype(np.int64)
    return binary_metrics(y_true, pred, prob)


def main() -> None:
    args = parse_args()
    require_sklearn()
    seed_root = resolve_path(args.seed_root, DEFAULT_SEED_ROOT)
    contest_root = resolve_path(args.contest_root, DEFAULT_CONTEST_ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    print("loading/building strict SEED features...")
    x_seed, y_seed, subjects, sessions, seed_meta = build_seed_strict_features(args, seed_root, output_dir)
    print(seed_meta)
    print("loading/building contest validation features...")
    x_val, y_val, val_spans = build_contest_validation_features(args, contest_root, output_dir)
    print(f"contest validation: {x_val.shape}, labels={np.bincount(y_val)}")

    folds = subject_folds(subjects, args.folds, args.random_seed)
    rows: list[dict[str, object]] = []
    seed_oof_by_model: dict[str, np.ndarray] = {name: np.zeros(len(y_seed), dtype=np.float32) for name in model_names}
    seed_oof_seen = np.zeros(len(y_seed), dtype=bool)

    for fold_idx, test_subjects in enumerate(folds, start=1):
        test_mask = np.isin(subjects, test_subjects)
        train_mask = ~test_mask
        seed_oof_seen[test_mask] = True
        print(f"fold {fold_idx}: train_seed_subjects={len(set(subjects[train_mask]))} test_seed_subjects={len(test_subjects)}", flush=True)
        for model_name in model_names:
            bundle = fit_one(model_name, x_seed[train_mask], y_seed[train_mask], args.random_seed + fold_idx * 100)
            p_train = predict(bundle, x_seed[train_mask])
            p_test = predict(bundle, x_seed[test_mask])
            threshold = float(best_threshold(y_seed[train_mask], p_train)["threshold"])
            metrics = score_with_threshold(y_seed[test_mask], p_test, threshold)
            oracle = best_threshold(y_seed[test_mask], p_test)
            seed_oof_by_model[model_name][test_mask] = p_test
            row = {
                "model": model_name,
                "fold": fold_idx,
                "seed_test_accuracy": metrics["accuracy"],
                "seed_test_balanced_accuracy": metrics["balanced_accuracy"],
                "seed_test_oracle_accuracy": oracle["accuracy"],
                "threshold_from_seed_train": threshold,
            }
            rows.append(row)
            print(f"  {model_name}: SEED-test acc={metrics['accuracy']:.4f} oracle={oracle['accuracy']:.4f}", flush=True)

    if not seed_oof_seen.all():
        raise RuntimeError("Some SEED samples were not covered by OOF folds")

    val_rows: list[dict[str, object]] = []
    for model_name in model_names:
        oof_prob = seed_oof_by_model[model_name]
        seed_oof_threshold = float(best_threshold(y_seed, oof_prob)["threshold"])
        final_bundle = fit_one(model_name, x_seed, y_seed, args.random_seed + 999)
        p_seed_all = predict(final_bundle, x_seed)
        p_val = predict(final_bundle, x_val)
        seed_train_metrics = score_with_threshold(y_seed, p_seed_all, seed_oof_threshold)
        val_metrics_seed_threshold = score_with_threshold(y_val, p_val, seed_oof_threshold)
        val_oracle = best_threshold(y_val, p_val)
        val_metrics_oracle = score_with_threshold(y_val, p_val, float(val_oracle["threshold"]))
        val_rows.append(
            {
                "model": model_name,
                "seed_oof_threshold": seed_oof_threshold,
                "seed_train_accuracy_at_oof_threshold": seed_train_metrics["accuracy"],
                "contest_validation_accuracy_seed_threshold": val_metrics_seed_threshold["accuracy"],
                "contest_validation_balanced_accuracy_seed_threshold": val_metrics_seed_threshold["balanced_accuracy"],
                "contest_validation_oracle_threshold": float(val_oracle["threshold"]),
                "contest_validation_oracle_accuracy": val_metrics_oracle["accuracy"],
                "contest_validation_oracle_balanced_accuracy": val_metrics_oracle["balanced_accuracy"],
            }
        )
        joblib.dump(final_bundle, output_dir / f"strict_seed_only_{model_name}.joblib")

    cv_df = pd.DataFrame(rows)
    val_df = pd.DataFrame(val_rows).sort_values("contest_validation_accuracy_seed_threshold", ascending=False)
    cv_summary = cv_df.groupby("model")[["seed_test_accuracy", "seed_test_balanced_accuracy", "seed_test_oracle_accuracy"]].agg(["mean", "std"]).reset_index()
    cv_summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in cv_summary.columns
    ]
    cv_summary = cv_summary.sort_values("seed_test_accuracy_mean", ascending=False)

    out_xlsx = output_dir / "strict_seed_only_report.xlsx"
    with pd.ExcelWriter(out_xlsx) as writer:
        cv_df.to_excel(writer, sheet_name="seed_cv_folds", index=False)
        cv_summary.to_excel(writer, sheet_name="seed_cv_summary", index=False)
        val_df.to_excel(writer, sheet_name="contest_validation", index=False)

    report = {
        "constraint": "Train/test only on SEED; contest training set used only as external validation.",
        "seed_meta": json.loads(seed_meta),
        "contest_validation_shape": list(x_val.shape),
        "seed_cv_summary": cv_summary.to_dict(orient="records"),
        "contest_validation": val_df.to_dict(orient="records"),
        "target_validation_accuracy": 0.80,
        "best_honest_validation_accuracy": float(val_df.iloc[0]["contest_validation_accuracy_seed_threshold"]),
        "best_validation_oracle_accuracy": float(val_df["contest_validation_oracle_accuracy"].max()),
        "report_path": str(out_xlsx),
    }
    out_json = output_dir / "strict_seed_only_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSEED CV summary:")
    print(cv_summary.to_string(index=False))
    print("\nContest validation:")
    print(val_df.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
