from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.contest_data import aggregate_trial_predictions
from seed_transfer.model import best_threshold, binary_metrics
from seed_transfer.paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEED_ROOT, ensure_dir, resolve_path
from seed_transfer.raw_trials import build_contest_raw_trial_cache, build_public_raw_trial_cache, build_seed_raw_trial_cache
from seed_transfer.torch_models import (
    EEGNet,
    EEGTrialDataset,
    coral_loss,
    fit_supervised,
    get_device,
    mmd_loss,
    predict_proba,
    set_torch_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEGNet with SEED/contest supervised loss plus CORAL or MMD domain adaptation.")
    parser.add_argument("--seed-root", type=str, default=None)
    parser.add_argument("--contest-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--domain-loss", choices=["coral", "mmd"], default="coral")
    parser.add_argument("--domain-weight", type=float, default=0.05)
    parser.add_argument("--seed-weight", type=float, default=0.3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


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


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def fit_domain_adapt(
    model: EEGNet,
    x_seed: np.ndarray,
    y_seed: np.ndarray,
    x_contest: np.ndarray,
    y_contest: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
):
    model.to(device)
    seed_loader = DataLoader(EEGTrialDataset(x_seed, y_seed), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    contest_loader = DataLoader(EEGTrialDataset(x_contest, y_contest), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    seed_iter = infinite_loader(seed_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    align_fn = coral_loss if args.domain_loss == "coral" else mmd_loss
    steps = max(len(seed_loader), len(contest_loader))
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, float]] = []
    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        losses = []
        for step, (xc, yc) in enumerate(contest_loader, start=1):
            xs, ys = next(seed_iter)
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xc = xc.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_seed, emb_seed = model(xs, return_embedding=True)
            logits_contest, emb_contest = model(xc, return_embedding=True)
            loss_seed = criterion(logits_seed, ys)
            loss_contest = criterion(logits_contest, yc)
            loss_align = align_fn(emb_seed, emb_contest)
            loss = args.seed_weight * loss_seed + loss_contest + args.domain_weight * loss_align
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if step >= steps:
                break
        probs = predict_proba(model, x_val, batch_size=args.batch_size, device=device)
        metrics = binary_metrics(y_val, (probs >= 0.5).astype(np.int64), probs)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_accuracy@0.5": metrics["accuracy"]}
        history.append(row)
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:03d} loss={row['loss']:.4f} val_acc@0.5={metrics['accuracy']:.4f}", flush=True)
    model.load_state_dict(best_state)
    return history


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
    seed = np.load(seed_cache, allow_pickle=True)
    contest = np.load(contest_cache, allow_pickle=True)
    public = np.load(public_cache, allow_pickle=True)
    x_seed, y_seed = seed["x"].astype(np.float32), seed["y"].astype(np.int64)
    x_contest, y_contest = contest["x"].astype(np.float32), contest["y"].astype(np.int64)
    subjects = contest["subjects"].astype(str)
    trial_ids = contest["trial_ids"].astype(np.int64)
    x_public = public["x"].astype(np.float32)
    public_subjects = public["subjects"].astype(str)
    public_trial_ids = public["trial_ids"].astype(np.int64)

    # Light supervised pretraining on SEED.
    base_model = EEGNet()
    print("pretraining on SEED...")
    fit_supervised(
        base_model,
        x_seed,
        y_seed,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    folds = folds_by_subject(subjects, args.folds, args.random_seed)
    oof_probs = np.zeros(len(y_contest), dtype=np.float32)
    rows: list[dict[str, object]] = []
    for fold_idx, val_subjects in enumerate(folds, start=1):
        val_mask = np.isin(subjects, val_subjects)
        train_mask = ~val_mask
        model = EEGNet()
        model.load_state_dict(base_model.state_dict())
        print(f"domain adapt fold {fold_idx}: train_subjects={len(set(subjects[train_mask]))} val_subjects={len(val_subjects)}")
        fit_domain_adapt(
            model,
            x_seed,
            y_seed,
            x_contest[train_mask],
            y_contest[train_mask],
            x_contest[val_mask],
            y_contest[val_mask],
            args=args,
            device=device,
        )
        p_train = predict_proba(model, x_contest[train_mask], batch_size=args.batch_size, device=device)
        p_val = predict_proba(model, x_contest[val_mask], batch_size=args.batch_size, device=device)
        threshold = float(best_threshold(y_contest[train_mask], p_train)["threshold"])
        metrics = binary_metrics(y_contest[val_mask], (p_val >= threshold).astype(np.int64), p_val)
        oracle = best_threshold(y_contest[val_mask], p_val)
        oof_probs[val_mask] = p_val
        row = {
            "fold": fold_idx,
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "oracle_accuracy": oracle["accuracy"],
        }
        rows.append(row)
        print(row)

    oof_threshold = float(best_threshold(y_contest, oof_probs)["threshold"])
    oof_metrics = binary_metrics(y_contest, (oof_probs >= oof_threshold).astype(np.int64), oof_probs)

    final_model = EEGNet()
    final_model.load_state_dict(base_model.state_dict())
    fit_domain_adapt(
        final_model,
        x_seed,
        y_seed,
        x_contest,
        y_contest,
        x_contest,
        y_contest,
        args=args,
        device=device,
    )
    torch.save(final_model.state_dict(), output_dir / f"eegnet_domain_{args.domain_loss}.pt")
    public_probs = predict_proba(final_model, x_public, batch_size=args.batch_size, device=device)
    public_rows = aggregate_trial_predictions(
        [(str(public_subjects[i]), int(public_trial_ids[i]), i, i + 1) for i in range(len(public_subjects))],
        public_probs,
        threshold=oof_threshold,
    )
    public_df = pd.DataFrame(public_rows)
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(output_dir / f"public_test_submission_eegnet_{args.domain_loss}.xlsx", index=False)
    public_df["probability"] = public_probs
    public_df.to_excel(output_dir / f"public_test_prediction_details_eegnet_{args.domain_loss}.xlsx", index=False)
    np.savez_compressed(
        output_dir / f"predictions_eegnet_{args.domain_loss}.npz",
        contest_oof=oof_probs,
        contest_y=y_contest,
        contest_subjects=subjects,
        contest_trial_ids=trial_ids,
        public_probs=public_probs,
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        threshold=np.array([oof_threshold], dtype=np.float32),
    )
    cv_df = pd.DataFrame(rows)
    cv_df.to_excel(output_dir / f"eegnet_domain_{args.domain_loss}_cv.xlsx", index=False)
    report = {
        "domain_loss": args.domain_loss,
        "domain_weight": args.domain_weight,
        "seed_weight": args.seed_weight,
        "cv_accuracy_mean": float(cv_df["accuracy"].mean()),
        "cv_accuracy_std": float(cv_df["accuracy"].std()),
        "oof_threshold": oof_threshold,
        "oof_metrics": oof_metrics,
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "args": vars(args),
    }
    (output_dir / f"eegnet_domain_{args.domain_loss}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
