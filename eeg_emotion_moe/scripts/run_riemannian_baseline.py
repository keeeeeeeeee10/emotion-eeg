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

from seed_transfer.contest_data import aggregate_trial_predictions
from seed_transfer.model import best_threshold, binary_metrics
from seed_transfer.paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path
from seed_transfer.raw_trials import build_contest_raw_trial_cache, build_public_raw_trial_cache
from seed_transfer.riemannian import riemann_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Riemannian log-covariance Logistic/SVM baseline.")
    parser.add_argument("--contest-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", type=str, default="logreg")
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def require_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return StandardScaler, LogisticRegression, SVC


def folds_by_subject(subjects: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.array(sorted(set(subjects.tolist())))
    hc = np.array([s for s in unique if str(s).startswith("HC")])
    dep = np.array([s for s in unique if str(s).startswith("DEP")])
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
    out: list[list[str]] = [[] for _ in range(folds)]
    for group in [hc, dep]:
        for i, subject in enumerate(group):
            out[i % folds].append(str(subject))
    return [np.array(sorted(fold)) for fold in out]


def class_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n / (2 * pos), n / (2 * neg)).astype(np.float32)


def get_features(raw_cache: Path, feat_cache: Path, refresh: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.load(raw_cache, allow_pickle=True)
    if feat_cache.exists() and not refresh:
        feat = np.load(feat_cache, allow_pickle=True)
        return feat["x"].astype(np.float32), {key: raw[key] for key in raw.files if key != "x"}
    x = riemann_features(raw["x"].astype(np.float32), fs=250.0)
    np.savez_compressed(feat_cache, x=x)
    return x, {key: raw[key] for key in raw.files if key != "x"}


def make_models(seed: int):
    _, LogisticRegression, SVC = require_sklearn()
    return {
        "logreg": LogisticRegression(C=0.5, max_iter=5000, solver="lbfgs", random_state=seed),
        "svm": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed + 1),
    }


def fit_bundle(model_name: str, x: np.ndarray, y: np.ndarray, seed: int):
    StandardScaler, _, _ = require_sklearn()
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    models = make_models(seed)
    if model_name == "ensemble":
        fitted = {}
        for name in ["logreg", "svm"]:
            model = deepcopy(models[name])
            try:
                model.fit(xs, y, sample_weight=class_weights(y))
            except TypeError:
                model.fit(xs, y)
            fitted[name] = model
        return {"scaler": scaler, "models": fitted, "model_name": model_name}
    model = deepcopy(models[model_name])
    try:
        model.fit(xs, y, sample_weight=class_weights(y))
    except TypeError:
        model.fit(xs, y)
    return {"scaler": scaler, "models": {model_name: model}, "model_name": model_name}


def predict(bundle: dict, x: np.ndarray) -> np.ndarray:
    xs = bundle["scaler"].transform(x)
    probs = [model.predict_proba(xs)[:, 1].astype(np.float32) for model in bundle["models"].values()]
    return np.mean(np.vstack(probs), axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    require_sklearn()
    contest_root = resolve_path(args.contest_root, DEFAULT_CONTEST_ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    contest_raw = build_contest_raw_trial_cache(contest_root=contest_root, output_dir=output_dir, refresh=args.refresh_cache)
    public_raw = build_public_raw_trial_cache(contest_root=contest_root, output_dir=output_dir, refresh=args.refresh_cache)
    x_contest, contest_meta = get_features(contest_raw, output_dir / "cache_contest_riemann_features.npz", args.refresh_features)
    x_public, public_meta = get_features(public_raw, output_dir / "cache_public_riemann_features.npz", args.refresh_features)
    y = contest_meta["y"].astype(np.int64)
    subjects = contest_meta["subjects"].astype(str)
    trial_ids = contest_meta["trial_ids"].astype(np.int64)
    public_subjects = public_meta["subjects"].astype(str)
    public_trial_ids = public_meta["trial_ids"].astype(np.int64)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    folds = folds_by_subject(subjects, args.folds, args.random_seed)
    rows: list[dict[str, object]] = []
    oof_by_model = {name: np.zeros(len(y), dtype=np.float32) for name in model_names}

    for fold_idx, test_subjects in enumerate(folds, start=1):
        test_mask = np.isin(subjects, test_subjects)
        train_mask = ~test_mask
        print(f"fold {fold_idx}: train={len(set(subjects[train_mask]))} val={len(test_subjects)}", flush=True)
        for model_name in model_names:
            bundle = fit_bundle(model_name, x_contest[train_mask], y[train_mask], args.random_seed + fold_idx * 100)
            p_train = predict(bundle, x_contest[train_mask])
            p_val = predict(bundle, x_contest[test_mask])
            threshold = float(best_threshold(y[train_mask], p_train)["threshold"])
            metrics = binary_metrics(y[test_mask], (p_val >= threshold).astype(np.int64), p_val)
            oracle = best_threshold(y[test_mask], p_val)
            oof_by_model[model_name][test_mask] = p_val
            row = {
                "fold": fold_idx,
                "model": model_name,
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "oracle_accuracy": oracle["accuracy"],
            }
            rows.append(row)
            print(f"  {model_name}: acc={metrics['accuracy']:.4f} oracle={oracle['accuracy']:.4f}", flush=True)

    cv_df = pd.DataFrame(rows)
    summary = cv_df.groupby("model")[["accuracy", "balanced_accuracy", "oracle_accuracy"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col) for col in summary.columns]
    summary = summary.sort_values("accuracy_mean", ascending=False)
    best_model = str(summary.iloc[0]["model"])
    final_bundle = fit_bundle(best_model, x_contest, y, args.random_seed + 999)
    p_all = predict(final_bundle, x_contest)
    threshold = float(best_threshold(y, p_all)["threshold"])
    p_public = predict(final_bundle, x_public)
    rows_public = aggregate_trial_predictions(
        [(str(public_subjects[i]), int(public_trial_ids[i]), i, i + 1) for i in range(len(public_subjects))],
        p_public,
        threshold=threshold,
    )
    public_df = pd.DataFrame(rows_public)
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(output_dir / "public_test_submission_riemannian.xlsx", index=False)
    public_df["probability"] = p_public
    public_df.to_excel(output_dir / "public_test_prediction_details_riemannian.xlsx", index=False)
    np.savez_compressed(
        output_dir / "predictions_riemannian.npz",
        contest_oof=oof_by_model[best_model],
        contest_y=y,
        contest_subjects=subjects,
        contest_trial_ids=trial_ids,
        public_probs=p_public,
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        threshold=np.array([threshold], dtype=np.float32),
        best_model=np.array([best_model]),
    )
    joblib.dump(final_bundle, output_dir / "riemannian_best_model.joblib")
    with pd.ExcelWriter(output_dir / "riemannian_cv_report.xlsx") as writer:
        cv_df.to_excel(writer, sheet_name="folds", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    report = {
        "best_model": best_model,
        "cv_summary": summary.to_dict(orient="records"),
        "threshold": threshold,
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
    }
    (output_dir / "riemannian_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
