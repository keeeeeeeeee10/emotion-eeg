from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.contest_data import aggregate_trial_predictions
from seed_transfer.model import best_threshold, binary_metrics
from seed_transfer.paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEED_ROOT, ensure_dir, resolve_path
from seed_transfer.raw_trials import (
    build_contest_raw_trial_cache,
    build_public_raw_trial_cache,
    build_seed_raw_trial_cache,
)
from seed_transfer.torch_models import EEGNet, fit_supervised, get_device, predict_proba, set_torch_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEGNet SEED pretraining + contest fine-tuning.")
    parser.add_argument("--seed-root", type=str, default=None)
    parser.add_argument("--contest-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--seed-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-seed", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def folds_by_subject(subjects: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.array(sorted(set(subjects.tolist())))
    hc = np.array([s for s in unique if str(s).startswith("HC")])
    dep = np.array([s for s in unique if str(s).startswith("DEP")])
    other = np.array([s for s in unique if not (str(s).startswith("HC") or str(s).startswith("DEP"))])
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
    rng.shuffle(other)
    out: list[list[str]] = [[] for _ in range(folds)]
    for group in [hc, dep, other]:
        for i, subject in enumerate(group):
            out[i % folds].append(str(subject))
    return [np.array(sorted(fold)) for fold in out]


def save_prediction_xlsx(path: Path, subjects: np.ndarray, trial_ids: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = aggregate_trial_predictions(
        [(str(subjects[i]), int(trial_ids[i]), i, i + 1) for i in range(len(subjects))],
        probs,
        threshold=threshold,
    )
    df = pd.DataFrame(rows)
    df[["user_id", "trial_id", "Emotion_label"]].to_excel(path, index=False)
    return df


def main() -> None:
    args = parse_args()
    set_torch_seed(args.random_seed)
    device = get_device()
    print(f"device={device}")
    seed_root = resolve_path(args.seed_root, DEFAULT_SEED_ROOT)
    contest_root = resolve_path(args.contest_root, DEFAULT_CONTEST_ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))

    seed_cache = build_seed_raw_trial_cache(seed_root=seed_root, output_dir=output_dir, refresh=args.refresh_cache)
    contest_cache = build_contest_raw_trial_cache(contest_root=contest_root, output_dir=output_dir, refresh=args.refresh_cache)
    public_cache = build_public_raw_trial_cache(contest_root=contest_root, output_dir=output_dir, refresh=args.refresh_cache)
    seed_data = np.load(seed_cache, allow_pickle=True)
    contest_data = np.load(contest_cache, allow_pickle=True)
    public_data = np.load(public_cache, allow_pickle=True)

    x_seed = seed_data["x"].astype(np.float32)
    y_seed = seed_data["y"].astype(np.int64)
    seed_subjects = seed_data["subjects"].astype(str)
    x_contest = contest_data["x"].astype(np.float32)
    y_contest = contest_data["y"].astype(np.int64)
    contest_subjects = contest_data["subjects"].astype(str)
    contest_trial_ids = contest_data["trial_ids"].astype(np.int64)
    x_public = public_data["x"].astype(np.float32)
    public_subjects = public_data["subjects"].astype(str)
    public_trial_ids = public_data["trial_ids"].astype(np.int64)

    # SEED pretraining with one subject-fold held out for sanity checking.
    seed_folds = folds_by_subject(seed_subjects, 5, args.random_seed)
    seed_test_subjects = seed_folds[0]
    seed_test_mask = np.isin(seed_subjects, seed_test_subjects)
    seed_train_mask = ~seed_test_mask
    pretrain_model = EEGNet()
    print("pretraining EEGNet on SEED...")
    seed_result = fit_supervised(
        pretrain_model,
        x_seed[seed_train_mask],
        y_seed[seed_train_mask],
        x_seed[seed_test_mask],
        y_seed[seed_test_mask],
        epochs=args.seed_epochs,
        batch_size=args.batch_size,
        lr=args.lr_seed,
        weight_decay=args.weight_decay,
        device=device,
    )
    seed_probs = predict_proba(pretrain_model, x_seed[seed_test_mask], batch_size=args.batch_size, device=device)
    seed_threshold = float(best_threshold(y_seed[seed_train_mask], predict_proba(pretrain_model, x_seed[seed_train_mask], batch_size=args.batch_size, device=device))["threshold"])
    seed_metrics = binary_metrics(y_seed[seed_test_mask], (seed_probs >= seed_threshold).astype(np.int64), seed_probs)
    torch.save(pretrain_model.state_dict(), output_dir / "eegnet_seed_pretrained.pt")

    folds = folds_by_subject(contest_subjects, args.folds, args.random_seed)
    cv_rows: list[dict[str, object]] = []
    oof_probs = np.zeros(len(y_contest), dtype=np.float32)
    for fold_idx, test_subjects in enumerate(folds, start=1):
        test_mask = np.isin(contest_subjects, test_subjects)
        train_mask = ~test_mask
        model = EEGNet()
        model.load_state_dict(seed_result.best_state)
        print(f"fine-tune fold {fold_idx}: train_subjects={len(set(contest_subjects[train_mask]))} val_subjects={len(test_subjects)}")
        fit_supervised(
            model,
            x_contest[train_mask],
            y_contest[train_mask],
            x_contest[test_mask],
            y_contest[test_mask],
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.lr_finetune,
            weight_decay=args.weight_decay,
            device=device,
        )
        p_train = predict_proba(model, x_contest[train_mask], batch_size=args.batch_size, device=device)
        p_val = predict_proba(model, x_contest[test_mask], batch_size=args.batch_size, device=device)
        threshold = float(best_threshold(y_contest[train_mask], p_train)["threshold"])
        metrics = binary_metrics(y_contest[test_mask], (p_val >= threshold).astype(np.int64), p_val)
        oracle = best_threshold(y_contest[test_mask], p_val)
        oof_probs[test_mask] = p_val
        row = {
            "fold": fold_idx,
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "oracle_threshold": oracle["threshold"],
            "oracle_accuracy": oracle["accuracy"],
        }
        cv_rows.append(row)
        print(row)

    oof_threshold = float(best_threshold(y_contest, oof_probs)["threshold"])
    oof_metrics = binary_metrics(y_contest, (oof_probs >= oof_threshold).astype(np.int64), oof_probs)

    final_model = EEGNet()
    final_model.load_state_dict(seed_result.best_state)
    print("training final EEGNet on all contest training data...")
    fit_supervised(
        final_model,
        x_contest,
        y_contest,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr_finetune,
        weight_decay=args.weight_decay,
        device=device,
    )
    torch.save(final_model.state_dict(), output_dir / "eegnet_finetuned.pt")
    public_probs = predict_proba(final_model, x_public, batch_size=args.batch_size, device=device)
    public_path = output_dir / "public_test_submission_eegnet.xlsx"
    public_detail = save_prediction_xlsx(public_path, public_subjects, public_trial_ids, public_probs, oof_threshold)
    public_detail["probability"] = public_probs
    public_detail.to_excel(output_dir / "public_test_prediction_details_eegnet.xlsx", index=False)
    np.savez_compressed(
        output_dir / "predictions_eegnet.npz",
        contest_oof=oof_probs,
        contest_y=y_contest,
        contest_subjects=contest_subjects,
        contest_trial_ids=contest_trial_ids,
        public_probs=public_probs,
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        threshold=np.array([oof_threshold], dtype=np.float32),
    )

    cv_df = pd.DataFrame(cv_rows)
    report = {
        "device": str(device),
        "seed_metrics": seed_metrics,
        "seed_best_accuracy": seed_result.best_accuracy,
        "contest_cv_accuracy_mean": float(cv_df["accuracy"].mean()),
        "contest_cv_accuracy_std": float(cv_df["accuracy"].std()),
        "contest_oof_threshold": oof_threshold,
        "contest_oof_metrics": oof_metrics,
        "public_positive_count": int((public_detail["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_detail["Emotion_label"] == 0).sum()),
        "args": vars(args),
    }
    (output_dir / "eegnet_pretrain_finetune_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cv_df.to_excel(output_dir / "eegnet_pretrain_finetune_cv.xlsx", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
