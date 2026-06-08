from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.model import binary_metrics
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


FIXED_FINAL_SOURCES = [
    ("predictions_eegnet.npz", 0.150),
    ("predictions_eegnet_mmd.npz", 0.050),
    ("predictions_riemannian.npz", 0.000),
    ("predictions_modma_softgate.npz", 0.200),
    ("predictions_disease_aware_ensemble.npz", 0.600),
    ("predictions_disease_pairwise.npz", 0.000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised contrastive MLP branch on trial-summary EEG features."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--outer-seeds", type=str, default="2026,2031,2042")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--norm-mode", type=str, default="subject_center", choices=["none", "subject_center", "subject_zscore"])
    parser.add_argument("--topk-train", type=int, default=20)
    parser.add_argument("--topk-public", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Install PyTorch before running this script.") from exc
    return torch, nn, F


def is_dep_subjects(subjects: np.ndarray) -> np.ndarray:
    return np.asarray([str(subject).upper().startswith("DEP") for subject in subjects], dtype=bool)


def subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(subjects.tolist())), dtype=str)
    dep = unique[is_dep_subjects(unique)]
    hc = unique[~is_dep_subjects(unique)]
    rng = np.random.default_rng(seed)
    rng.shuffle(hc)
    rng.shuffle(dep)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for group in [hc, dep]:
        for idx, subject in enumerate(group):
            folds[idx % n_folds].append(str(subject))
    return [np.asarray(sorted(fold), dtype=str) for fold in folds]


def topk_by_subject(scores: np.ndarray, subjects: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        pred[idx[np.argsort(scores[idx])[::-1][:k]]] = 1
    return pred


def rank_by_subject(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        if len(idx) == 1:
            out[idx] = 0.5
            continue
        order = np.argsort(scores[idx])
        local = np.empty(len(idx), dtype=np.float32)
        local[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        out[idx] = local
    return out


def score_with_groups(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> dict[str, float]:
    pred = topk_by_subject(scores, subjects, k)
    dep_mask = is_dep_subjects(subjects)
    metrics = binary_metrics(y, pred, scores)
    metrics["hc_accuracy"] = float((pred[~dep_mask] == y[~dep_mask]).mean())
    metrics["dep_accuracy"] = float((pred[dep_mask] == y[dep_mask]).mean())
    return metrics


def video_and_offset(trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    video_id = np.where(trial_ids <= 20, (trial_ids - 1) // 5 + 1, (trial_ids - 21) // 5 + 5).astype(np.int64)
    offset = np.where(trial_ids <= 20, (trial_ids - 1) % 5, (trial_ids - 21) % 5).astype(np.int64)
    return video_id, offset


def video_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, trial_ids: np.ndarray, *, agg: str) -> dict[str, float]:
    video_id, _ = video_and_offset(trial_ids)
    rows: list[tuple[str, int, int, float]] = []
    for subject in sorted(set(subjects.tolist())):
        for vid in range(1, 9):
            idx = np.where((subjects == subject) & (video_id == vid))[0]
            block = scores[idx]
            if agg == "mean":
                score = float(block.mean())
            elif agg == "median":
                score = float(np.median(block))
            elif agg == "q75":
                score = float(np.quantile(block, 0.75))
            else:
                raise ValueError(agg)
            rows.append((subject, vid, int(round(float(y[idx].mean()))), score))
    v_subjects = np.asarray([row[0] for row in rows], dtype=str)
    v_y = np.asarray([row[2] for row in rows], dtype=np.int64)
    v_scores = rank_by_subject(np.asarray([row[3] for row in rows], dtype=np.float32), v_subjects)
    return score_with_groups(v_y, v_scores, v_subjects, 4)


def offset_proxy_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, trial_ids: np.ndarray) -> dict[str, float]:
    _, offsets = video_and_offset(trial_ids)
    rows = []
    for offset in range(5):
        mask = offsets == offset
        rows.append(score_with_groups(y[mask], scores[mask], subjects[mask], 4))
    return {
        "offset_accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
        "offset_accuracy_min": float(np.min([row["accuracy"] for row in rows])),
        "offset_hc_mean": float(np.mean([row["hc_accuracy"] for row in rows])),
        "offset_dep_mean": float(np.mean([row["dep_accuracy"] for row in rows])),
    }


def normalize_by_subject(x: np.ndarray, subjects: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32, copy=True)
    out = np.empty_like(x, dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        block = x[idx]
        mean = block.mean(axis=0, keepdims=True)
        if mode == "subject_center":
            out[idx] = block - mean
        elif mode == "subject_zscore":
            std = block.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            out[idx] = (block - mean) / std
        else:
            raise ValueError(mode)
    return out.astype(np.float32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    scale = x.std(axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def transform_standardizer(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32)


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg)).astype(np.float32)


def load_cached_features(output_dir: Path) -> dict[str, np.ndarray]:
    contest_path = output_dir / "cache_contest_trial_summary.npz"
    public_path = output_dir / "cache_public_trial_summary_from_raw.npz"
    if not contest_path.exists():
        raise FileNotFoundError(f"Missing contest cache: {contest_path}")
    if not public_path.exists():
        raise FileNotFoundError(f"Missing public cache: {public_path}")
    contest = np.load(contest_path, allow_pickle=True)
    public = np.load(public_path, allow_pickle=True)
    spans = [tuple(row) for row in contest["spans"].tolist()]
    return {
        "x": contest["x"].astype(np.float32),
        "y": contest["y"].astype(np.int64),
        "subjects": contest["subjects"].astype(str),
        "trial_ids": np.asarray([int(row[1]) for row in spans], dtype=np.int64),
        "public_x": public["x"].astype(np.float32),
        "public_subjects": public["subjects"].astype(str),
        "public_trial_ids": public["trial_ids"].astype(np.int64),
    }


def load_fixed_final(output_dir: Path) -> dict[str, np.ndarray]:
    loaded = []
    for filename, weight in FIXED_FINAL_SOURCES:
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing fixed-final source: {path}")
        loaded.append((filename, float(weight), np.load(path, allow_pickle=True)))
    y = loaded[0][2]["contest_y"].astype(np.int64)
    subjects = loaded[0][2]["contest_subjects"].astype(str)
    trial_ids = loaded[0][2]["contest_trial_ids"].astype(np.int64)
    public_subjects = loaded[0][2]["public_subjects"].astype(str)
    public_trial_ids = loaded[0][2]["public_trial_ids"].astype(np.int64)
    scores = np.zeros(len(y), dtype=np.float32)
    for filename, weight, data in loaded:
        if not np.array_equal(y, data["contest_y"].astype(np.int64)):
            raise ValueError(f"contest_y mismatch in {filename}")
        scores += float(weight) * data["contest_oof"].astype(np.float32)
    return {
        "contest_scores": rank_by_subject(scores, subjects),
        "contest_y": y,
        "contest_subjects": subjects,
        "contest_trial_ids": trial_ids,
        "public_subjects": public_subjects,
        "public_trial_ids": public_trial_ids,
    }


def make_model_class():
    torch, nn, F = require_torch()

    class SupConMLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.projector = nn.Linear(hidden_dim, embed_dim)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.encoder(x)
            z = F.normalize(self.projector(h), dim=1)
            logits = self.classifier(h).squeeze(1)
            return logits, z

    return SupConMLP


def supervised_contrastive_loss(z, labels, *, temperature: float):
    torch, _, _ = require_torch()
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    logits = torch.matmul(z, z.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = mask.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return torch.zeros((), dtype=z.dtype, device=z.device)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def train_epoch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    embed_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    supcon_weight: float,
    temperature: float,
) -> np.ndarray:
    torch, nn, _ = require_torch()
    Model = make_model_class()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    model = Model(x_train.shape[1], hidden_dim, embed_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
    w_t = torch.as_tensor(class_balanced_weights(y_train), dtype=torch.float32, device=device)
    for _ in range(max(int(epochs), 1)):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_t[idx]
            yb = y_t[idx]
            wb = w_t[idx]
            opt.zero_grad(set_to_none=True)
            logits, z = model(xb)
            bce_raw = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce_raw * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            con = supervised_contrastive_loss(z, yb.long(), temperature=temperature)
            loss = bce + float(supcon_weight) * con
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(x_eval, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_dim: int,
    embed_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    supcon_weight: float,
    temperature: float,
) -> tuple[np.ndarray, dict[str, object]]:
    torch, nn, _ = require_torch()
    Model = make_model_class()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inner = subject_folds(subjects_train, 5, seed + 31)
    dev_subjects = inner[0]
    dev_mask = np.isin(subjects_train, dev_subjects)
    core_mask = ~dev_mask
    if int(core_mask.sum()) < 40 or int(dev_mask.sum()) < 20:
        core_mask = np.ones(len(y_train), dtype=bool)
        dev_mask = np.ones(len(y_train), dtype=bool)

    mean, scale = fit_standardizer(x_train[core_mask])
    x_core = transform_standardizer(x_train[core_mask], mean, scale)
    x_dev = transform_standardizer(x_train[dev_mask], mean, scale)
    y_core = y_train[core_mask]
    y_dev = y_train[dev_mask]
    subjects_dev = subjects_train[dev_mask]

    device = torch.device("cpu")
    model = Model(x_train.shape[1], hidden_dim, embed_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_core_t = torch.as_tensor(x_core, dtype=torch.float32, device=device)
    y_core_t = torch.as_tensor(y_core.astype(np.float32), dtype=torch.float32, device=device)
    w_core_t = torch.as_tensor(class_balanced_weights(y_core), dtype=torch.float32, device=device)
    x_dev_t = torch.as_tensor(x_dev, dtype=torch.float32, device=device)
    best_epoch = 1
    best_acc = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y_core))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_core_t[idx]
            yb = y_core_t[idx]
            wb = w_core_t[idx]
            opt.zero_grad(set_to_none=True)
            logits, z = model(xb)
            bce_raw = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce_raw * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            con = supervised_contrastive_loss(z, yb.long(), temperature=temperature)
            loss = bce + float(supcon_weight) * con
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_dev_t)
            dev_scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        dev_metrics = score_with_groups(y_dev, dev_scores, subjects_dev, 20)
        history.append({"epoch": epoch, "dev_accuracy": dev_metrics["accuracy"]})
        if dev_metrics["accuracy"] > best_acc:
            best_acc = float(dev_metrics["accuracy"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Refit on all outer-train subjects using the inner-selected horizon.
    mean_all, scale_all = fit_standardizer(x_train)
    scores = train_epoch_model(
        transform_standardizer(x_train, mean_all, scale_all),
        y_train,
        transform_standardizer(x_eval, mean_all, scale_all),
        seed=seed + 991,
        epochs=best_epoch,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        supcon_weight=supcon_weight,
        temperature=temperature,
    )
    return scores, {
        "best_epoch": best_epoch,
        "best_dev_accuracy": best_acc,
        "epochs_ran": len(history),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    torch, _, _ = require_torch()
    torch.set_num_threads(max(int(args.torch_threads), 1))
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "main"
    report_dir = ensure_dir(output_dir / f"supcon_mlp_{safe_tag}")
    data = load_cached_features(output_dir)
    fixed_final = load_fixed_final(output_dir)
    x = normalize_by_subject(data["x"], data["subjects"], args.norm_mode)
    public_x = normalize_by_subject(data["public_x"], data["public_subjects"], args.norm_mode)
    y = data["y"]
    subjects = data["subjects"]
    trial_ids = data["trial_ids"]
    if not np.array_equal(y, fixed_final["contest_y"]) or not np.array_equal(subjects, fixed_final["contest_subjects"]):
        raise ValueError("Fixed final predictions are not aligned with summary cache")
    seeds = [int(part.strip()) for part in args.outer_seeds.split(",") if part.strip()]
    print(
        f"loaded x={x.shape}, seeds={seeds}, hidden={args.hidden_dim}, embed={args.embed_dim}, "
        f"supcon_weight={args.supcon_weight}",
        flush=True,
    )
    seed_rows = []
    fold_rows = []
    seed_oof: dict[int, np.ndarray] = {}
    best_epochs = []
    for seed in seeds:
        oof = np.zeros(len(y), dtype=np.float32)
        print(f"\nouter seed {seed}", flush=True)
        for fold_idx, val_subjects in enumerate(subject_folds(subjects, args.folds, seed), start=1):
            val_mask = np.isin(subjects, val_subjects)
            train_mask = ~val_mask
            scores, info = train_one_fold(
                x[train_mask],
                y[train_mask],
                subjects[train_mask],
                x[val_mask],
                seed=seed + fold_idx * 1000,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                supcon_weight=args.supcon_weight,
                temperature=args.temperature,
            )
            oof[val_mask] = scores
            best_epochs.append(int(info["best_epoch"]))
            metrics = score_with_groups(y[val_mask], scores, subjects[val_mask], args.topk_train)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold_idx,
                    "best_epoch": info["best_epoch"],
                    "best_dev_accuracy": info["best_dev_accuracy"],
                    "epochs_ran": info["epochs_ran"],
                    **metrics,
                }
            )
            print(
                f"  fold {fold_idx}: val={metrics['accuracy']:.4f} HC={metrics['hc_accuracy']:.4f} "
                f"DEP={metrics['dep_accuracy']:.4f} best_epoch={info['best_epoch']}",
                flush=True,
            )
        seed_oof[seed] = oof
        metrics = score_with_groups(y, oof, subjects, args.topk_train)
        seed_rows.append({"seed": seed, "variant": "supcon_mlp", **metrics})
        print(
            f"seed {seed}: {metrics['accuracy']:.4f} HC={metrics['hc_accuracy']:.4f} DEP={metrics['dep_accuracy']:.4f}",
            flush=True,
        )

    mean_scores = np.mean(np.vstack([seed_oof[s] for s in seeds]), axis=0).astype(np.float32)
    mean_metrics = score_with_groups(y, mean_scores, subjects, args.topk_train)
    video_rows = []
    for agg in ["mean", "median", "q75"]:
        video_rows.append({"variant": "supcon_mlp_mean", "agg": agg, **video_metrics(y, mean_scores, subjects, trial_ids, agg=agg)})
        video_rows.append({"variant": "fixed_final", "agg": agg, **video_metrics(y, fixed_final["contest_scores"], subjects, trial_ids, agg=agg)})
    offset_metrics = offset_proxy_metrics(y, mean_scores, subjects, trial_ids)
    fixed_offset_metrics = offset_proxy_metrics(y, fixed_final["contest_scores"], subjects, trial_ids)

    blend_rows = []
    supcon_rank = rank_by_subject(mean_scores, subjects)
    for alpha in np.linspace(0.0, 0.3, 7, dtype=np.float32):
        blend = (1.0 - float(alpha)) * fixed_final["contest_scores"] + float(alpha) * supcon_rank
        blend_rows.append({"alpha": float(alpha), **score_with_groups(y, blend, subjects, args.topk_train)})
    blend_df = pd.DataFrame(blend_rows).sort_values(["accuracy", "dep_accuracy"], ascending=False)

    full_epochs = int(np.median(best_epochs)) if best_epochs else max(1, args.epochs // 3)
    mean_all, scale_all = fit_standardizer(x)
    public_parts = []
    for seed in seeds:
        public_parts.append(
            train_epoch_model(
                transform_standardizer(x, mean_all, scale_all),
                y,
                transform_standardizer(public_x, mean_all, scale_all),
                seed=seed + 9000,
                epochs=full_epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                supcon_weight=args.supcon_weight,
                temperature=args.temperature,
            )
        )
    public_scores = np.mean(np.vstack(public_parts), axis=0).astype(np.float32)
    public_rank = rank_by_subject(public_scores, data["public_subjects"])
    public_labels = topk_by_subject(public_scores, data["public_subjects"], args.topk_public)
    public_df = pd.DataFrame(
        {
            "user_id": data["public_subjects"],
            "trial_id": data["public_trial_ids"],
            "score": public_scores,
            "rank_probability": public_rank,
            "Emotion_label": public_labels,
        }
    )
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(
        report_dir / f"public_test_submission_supcon_mlp_{safe_tag}_top4.xlsx",
        index=False,
    )
    public_df.to_excel(report_dir / f"public_test_prediction_details_supcon_mlp_{safe_tag}_top4.xlsx", index=False)

    np.savez_compressed(
        output_dir / f"predictions_supcon_mlp_{safe_tag}.npz",
        contest_oof=rank_by_subject(mean_scores, subjects).astype(np.float32),
        contest_raw_scores=mean_scores.astype(np.float32),
        contest_y=y,
        contest_subjects=subjects,
        contest_trial_ids=trial_ids,
        public_probs=public_rank.astype(np.float32),
        public_raw_scores=public_scores.astype(np.float32),
        public_subjects=data["public_subjects"],
        public_trial_ids=data["public_trial_ids"],
        threshold=np.array([0.5], dtype=np.float32),
        seed_oof=np.vstack([seed_oof[s] for s in seeds]).astype(np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
    )
    seed_df = pd.DataFrame(seed_rows)
    fold_df = pd.DataFrame(fold_rows)
    mean_df = pd.DataFrame([{"variant": "supcon_mlp_mean", **mean_metrics}])
    video_df = pd.DataFrame(video_rows)
    with pd.ExcelWriter(report_dir / f"supcon_mlp_{safe_tag}_report.xlsx") as writer:
        seed_df.to_excel(writer, sheet_name="seed_metrics", index=False)
        mean_df.to_excel(writer, sheet_name="mean_metrics", index=False)
        fold_df.to_excel(writer, sheet_name="fold_details", index=False)
        blend_df.to_excel(writer, sheet_name="blend_with_final", index=False)
        video_df.to_excel(writer, sheet_name="video_metrics", index=False)
        pd.DataFrame([{"variant": "supcon_mlp_mean", **offset_metrics}, {"variant": "fixed_final", **fixed_offset_metrics}]).to_excel(
            writer,
            sheet_name="offset_proxy_metrics",
            index=False,
        )
        public_df.to_excel(writer, sheet_name="public_top4", index=False)

    report = {
        "fixed_final_chunk_metrics": score_with_groups(y, fixed_final["contest_scores"], subjects, args.topk_train),
        "supcon_mean_metrics": mean_metrics,
        "video_metrics": video_df.to_dict(orient="records"),
        "offset_proxy_metrics": {
            "supcon_mlp_mean": offset_metrics,
            "fixed_final": fixed_offset_metrics,
        },
        "best_blend_with_final": blend_df.head(10).to_dict(orient="records"),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "full_public_epochs": full_epochs,
        "outputs": {
            "predictions": str(output_dir / f"predictions_supcon_mlp_{safe_tag}.npz"),
            "report": str(report_dir / f"supcon_mlp_{safe_tag}_report.xlsx"),
            "submission": str(report_dir / f"public_test_submission_supcon_mlp_{safe_tag}_top4.xlsx"),
        },
        "args": vars(args),
    }
    (report_dir / f"supcon_mlp_{safe_tag}_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
