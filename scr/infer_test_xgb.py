from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from features import extract_de_features, extract_psd_features


TEST_DIR = Path("data/test")
MODEL_PATH = Path("output/models/xgb_psd_de.pkl")
SCALER_PATH = Path("output/models/scaler_psd_de.pkl")
OUTPUT_XLSX = Path("output/submission_xgb_psd_de.xlsx")
OUTPUT_XLSX_TOP4 = Path("output/submission_xgb_psd_de_top4.xlsx")
OUTPUT_DEBUG_XLSX = Path("output/submission_xgb_psd_de_debug.xlsx")

N_CHANNELS = 30
TRIAL_SAMPLES = 2500
N_TRIALS = 8
TOTAL_SAMPLES = TRIAL_SAMPLES * N_TRIALS  # 20000


def _load_mat_test(mat_path: Path) -> dict[str, Any]:
    return loadmat(mat_path, squeeze_me=False, struct_as_record=False)


def _pick_eeg_array(data: dict[str, Any], mat_path: Path) -> np.ndarray:
    # 优先使用常见测试变量名
    preferred_keys = ("test_eeg_c", "EEG_data", "eeg", "data")
    for k in preferred_keys:
        if k in data and isinstance(data[k], np.ndarray):
            arr = np.asarray(data[k])
            if arr.ndim == 2:
                return arr

    # 兜底：找第一个二维数组
    for k, v in data.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return np.asarray(v)

    raise ValueError(f"{mat_path} has no 2D EEG array variable")


def _to_channel_first_30x20000(arr: np.ndarray, mat_path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{mat_path} EEG must be 2D, got {arr.shape}")

    if arr.shape == (N_CHANNELS, TOTAL_SAMPLES):
        return arr
    if arr.shape == (TOTAL_SAMPLES, N_CHANNELS):
        return arr.T
    raise ValueError(
        f"{mat_path} bad EEG shape={arr.shape}, expected {(N_CHANNELS, TOTAL_SAMPLES)} "
        f"or {(TOTAL_SAMPLES, N_CHANNELS)}"
    )


def _split_trials(eeg: np.ndarray) -> np.ndarray:
    # eeg: [30, 20000] -> [8, 30, 2500]
    trials = []
    for i in range(N_TRIALS):
        s = i * TRIAL_SAMPLES
        e = (i + 1) * TRIAL_SAMPLES
        trial = eeg[:, s:e]
        if trial.shape != (N_CHANNELS, TRIAL_SAMPLES):
            raise ValueError(f"Bad trial shape at trial={i+1}: {trial.shape}")
        trials.append(trial.astype(np.float32, copy=False))
    return np.stack(trials, axis=0)


def _zscore_per_trial_channel(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    return ((x - mean) / (std + 1e-6)).astype(np.float32, copy=False)


def extract_psd_de_features(X_trials: np.ndarray) -> np.ndarray:
    X_psd = extract_psd_features(X_trials)  # [8, 150]
    X_de = extract_de_features(X_trials)  # [8, 150]
    X = np.concatenate([X_psd, X_de], axis=1).astype(np.float32, copy=False)  # [8, 300]
    return X


def natural_key(path: Path) -> int | str:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else path.stem


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")

    test_files = sorted(TEST_DIR.glob("*.mat"), key=natural_key)
    if not test_files:
        raise FileNotFoundError(f"No test .mat files found under {TEST_DIR}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    rows: list[dict[str, int | str]] = []
    rows_debug: list[dict[str, int | float | str]] = []
    print(f"Found test files: {len(test_files)}")

    for mat_path in test_files:
        user_id = mat_path.stem
        mat = _load_mat_test(mat_path)
        eeg_raw = _pick_eeg_array(mat, mat_path)
        eeg = _to_channel_first_30x20000(eeg_raw, mat_path)
        X_trials = _zscore_per_trial_channel(_split_trials(eeg))  # [8, 30, 2500]

        X_feat = extract_psd_de_features(X_trials)  # [8, 300]
        X_scaled = scaler.transform(X_feat)
        probs = model.predict_proba(X_scaled)[:, 1]

        # 每个测试被试固定 8 个 trial，其中 4 个 positive，4 个 neutral
        labels = np.zeros_like(probs, dtype=int)
        top4_idx = np.argsort(probs)[-4:]
        labels[top4_idx] = 1

        if labels.shape[0] != N_TRIALS:
            raise RuntimeError(f"{mat_path} predicted trials != 8, got {labels.shape[0]}")

        print(f"{user_id} probs: {np.round(probs, 4)}")
        print(f"{user_id} top4 positive trial index: {np.sort(top4_idx + 1)}")
        print(f"{user_id} labels: {labels}")

        for trial_id, (prob, label) in enumerate(zip(probs, labels), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "trial_id": int(trial_id),
                    "Emotion_label": int(label),
                }
            )
            rows_debug.append(
                {
                    "user_id": user_id,
                    "trial_id": int(trial_id),
                    "prob_positive": float(prob),
                    "Emotion_label": int(label),
                }
            )

        print(f"{user_id}: predicted {N_TRIALS} trials")

    df = pd.DataFrame(rows, columns=["user_id", "trial_id", "Emotion_label"])
    df_debug = pd.DataFrame(
        rows_debug, columns=["user_id", "trial_id", "prob_positive", "Emotion_label"]
    )
    df_submit = df_debug[["user_id", "trial_id", "Emotion_label"]]

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    # 兼容旧文件名
    df.to_excel(OUTPUT_XLSX, index=False)
    # 推荐正式提交文件（top4策略）
    df_submit.to_excel(OUTPUT_XLSX_TOP4, index=False)
    # 调试文件（含概率）
    df_debug.to_excel(OUTPUT_DEBUG_XLSX, index=False)

    print(f"Saved submission (legacy name): {OUTPUT_XLSX}")
    print(f"Saved submission (top4): {OUTPUT_XLSX_TOP4}")
    print(f"Saved debug file: {OUTPUT_DEBUG_XLSX}")
    print(f"Rows: {len(df_submit)}")
    print(df_submit.head(10))


if __name__ == "__main__":
    main()
