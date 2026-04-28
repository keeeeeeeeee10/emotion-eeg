from __future__ import annotations

import os
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")


def use_gpu() -> bool:
    mode = os.environ.get("EEG_USE_GPU", "auto").strip().lower()
    if mode in {"0", "false", "no", "cpu"}:
        return False
    if mode in {"1", "true", "yes", "gpu", "cuda"}:
        return True

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return shutil.which("nvidia-smi") is not None


def xgb_device_params() -> dict[str, str]:
    if use_gpu():
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist", "device": "cpu"}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_path = FEATURE_DIR / "X_psd_de.npy"
    y_path = FEATURE_DIR / "y.npy"
    g_path = FEATURE_DIR / "groups.npy"

    if not x_path.exists() or not y_path.exists() or not g_path.exists():
        raise FileNotFoundError(
            "Missing feature files. Please run build_dataset.py and make_features.py first."
        )

    X = np.load(x_path)
    y = np.load(y_path)
    groups = np.load(g_path)
    return X, y, groups


def sanity_check(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"groups: {groups.shape}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N, D], got {X.shape}")
    if y.ndim != 1 or groups.ndim != 1:
        raise ValueError(f"y/groups must be 1D, got y={y.shape}, groups={groups.shape}")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, groups length mismatch")

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    label_vals = np.unique(y)
    subject_ids, subject_counts = np.unique(groups, return_counts=True)

    print(f"nan: {nan_count}")
    print(f"inf: {inf_count}")
    print(f"label distribution: {np.bincount(y.astype(int))}")
    print(f"num subjects: {len(subject_ids)}")
    print(
        "samples per subject (min/median/max): "
        f"{subject_counts.min()}/{int(np.median(subject_counts))}/{subject_counts.max()}"
    )

    if nan_count > 0 or inf_count > 0:
        raise ValueError("X contains NaN or Inf.")
    if set(label_vals.tolist()) != {0, 1}:
        raise ValueError(f"y must be binary {{0,1}}, got {label_vals}")
    if len(subject_ids) != 60:
        print(f"[WARN] Expected 60 subjects, got {len(subject_ids)}")
    if not np.all(subject_counts == 40):
        print("[WARN] Not all subjects have exactly 40 samples.")


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        **xgb_device_params(),
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def run_group_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> None:
    gkf = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []
    fold_f1s: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        g_train, g_val = groups[train_idx], groups[val_idx]

        train_subjects = set(g_train.tolist())
        val_subjects = set(g_val.tolist())
        overlap = train_subjects & val_subjects

        print("\n" + "=" * 70)
        print(f"Fold {fold}")
        print("=" * 70)
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
        print(f"y_train: {np.bincount(y_train.astype(int))}, y_val: {np.bincount(y_val.astype(int))}")
        print(f"train subjects: {len(train_subjects)}, val subjects: {len(val_subjects)}")
        print(f"Subject leakage: {overlap}")
        if overlap:
            raise RuntimeError(f"Detected subject leakage at fold {fold}: {overlap}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = build_model()
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_val_scaled)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

        fold_accs.append(acc)
        fold_f1s.append(f1)

        print(f"ACC: {acc:.4f}")
        print(f"F1 : {f1:.4f}")
        print(f"Pred label distribution: {np.bincount(y_pred.astype(int), minlength=2)}")
        print("Confusion matrix:")
        print(cm)
        print("Classification report:")
        print(classification_report(y_val, y_pred, digits=4))

    print("\n" + "#" * 70)
    print("Final Cross-Subject CV Result")
    print("#" * 70)
    print(f"ACC: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"F1 : {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")


def train_and_save_final(X: np.ndarray, y: np.ndarray) -> None:
    print("\nTraining final model on all training samples...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = build_model()
    clf.fit(X_scaled, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgb_psd_de.pkl"
    scaler_path = MODEL_DIR / "scaler_psd_de.pkl"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved model : {model_path}")
    print(f"Saved scaler: {scaler_path}")


def main() -> None:
    X, y, groups = load_data()
    y = y.astype(np.int64, copy=False)
    groups = groups.astype(np.int64, copy=False)

    sanity_check(X, y, groups)
    print(f"GPU mode: {'cuda' if use_gpu() else 'cpu'} for XGBoost")
    run_group_cv(X, y, groups, n_splits=5)
    train_and_save_final(X, y)


if __name__ == "__main__":
    main()
