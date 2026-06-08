from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.model import binary_metrics
from seed_transfer.paths import DEFAULT_OUTPUT_DIR, ensure_dir, resolve_path


DEFAULT_SOURCES = (
    "predictions_eegnet.npz,"
    "predictions_eegnet_coral.npz,"
    "predictions_eegnet_mmd.npz,"
    "predictions_riemannian.npz,"
    "predictions_sklearn.npz,"
    "predictions_seed_pairwise_ranking.npz,"
    "predictions_modma_softgate.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search HC/DEP-specific ensemble weights and blend them with MODMA p_DEP gate."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sources", type=str, default=DEFAULT_SOURCES)
    parser.add_argument("--gate-source", type=str, default="predictions_modma_softgate.npz")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--gate-alphas", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--score-spaces", type=str, default="rank,raw")
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    return {
        "path": path.name,
        "contest_oof": data["contest_oof"].astype(np.float32),
        "contest_y": data["contest_y"].astype(np.int64),
        "contest_subjects": data["contest_subjects"].astype(str),
        "contest_trial_ids": data["contest_trial_ids"].astype(np.int64),
        "public_probs": data["public_probs"].astype(np.float32),
        "public_subjects": data["public_subjects"].astype(str),
        "public_trial_ids": data["public_trial_ids"].astype(np.int64),
    }


def load_gate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "gate_oof" not in data.files or "public_gate" not in data.files:
        raise KeyError(f"{path.name} must contain gate_oof and public_gate")
    return data["gate_oof"].astype(np.float32), data["public_gate"].astype(np.float32)


def is_dep_subject(subjects: np.ndarray) -> np.ndarray:
    return np.asarray([str(s).upper().startswith("DEP") for s in subjects], dtype=bool)


def topk_by_subject(scores: np.ndarray, subjects: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=np.int64)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        pred[order[:k]] = 1
    return pred


def rank_by_subject(scores: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float32)
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        local = scores[idx]
        if len(idx) == 1:
            out[idx] = 0.5
            continue
        order = np.argsort(local)
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
        out[idx] = ranks
    return out


def subset_topk_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = topk_by_subject(scores[mask], subjects[mask], 20)
    return binary_metrics(y[mask], pred, scores[mask])


def topk_metrics(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> dict[str, float]:
    pred = topk_by_subject(scores, subjects, k)
    return binary_metrics(y, pred, scores)


def simplex_weights(n_models: int, step: float):
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-6:
        raise ValueError("--grid-step must divide 1.0")

    def rec(remaining: int, slots: int):
        if slots == 1:
            yield [remaining]
            return
        for value in range(remaining + 1):
            for tail in rec(remaining - value, slots - 1):
                yield [value] + tail

    for counts in rec(units, n_models):
        yield np.asarray(counts, dtype=np.float32) / units


def weighted_sum(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(matrix * weights.reshape(1, -1), axis=1).astype(np.float32)


def search_group_weights(
    matrix: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
    *,
    step: float,
    group_name: str,
    score_space: str,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_weights: np.ndarray | None = None
    for weights in simplex_weights(matrix.shape[1], step):
        scores = weighted_sum(matrix, weights)
        metrics = subset_topk_metrics(y, scores, subjects, mask)
        row = {
            "group": group_name,
            "score_space": score_space,
            "weights": ",".join(f"{v:.3f}" for v in weights),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        rows.append(row)
        if best_row is None or row["accuracy"] > float(best_row["accuracy"]):
            best_row = row
            best_weights = weights.copy()
    assert best_row is not None and best_weights is not None
    return best_weights, best_row, rows


def by_subject_accuracy(y: np.ndarray, scores: np.ndarray, subjects: np.ndarray, k: int) -> pd.DataFrame:
    pred = topk_by_subject(scores, subjects, k)
    rows = []
    for subject in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subject)[0]
        rows.append(
            {
                "subject": subject,
                "type": "DEP" if str(subject).upper().startswith("DEP") else "HC",
                "accuracy": float((pred[idx] == y[idx]).mean()),
                "positive_mean_score": float(scores[idx][y[idx] == 1].mean()),
                "neutral_mean_score": float(scores[idx][y[idx] == 0].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    source_paths = [output_dir / part.strip() for part in args.sources.split(",") if part.strip()]
    preds = [load_prediction(path) for path in source_paths]
    gate_oof, public_gate = load_gate(output_dir / args.gate_source)

    y = preds[0]["contest_y"]  # type: ignore[index]
    contest_subjects = preds[0]["contest_subjects"]  # type: ignore[index]
    contest_trial_ids = preds[0]["contest_trial_ids"]  # type: ignore[index]
    public_subjects = preds[0]["public_subjects"]  # type: ignore[index]
    public_trial_ids = preds[0]["public_trial_ids"]  # type: ignore[index]
    for pred in preds[1:]:
        if not np.array_equal(y, pred["contest_y"]):
            raise ValueError(f"contest_y mismatch: {pred['path']}")
        if not np.array_equal(contest_subjects, pred["contest_subjects"]):
            raise ValueError(f"contest_subjects mismatch: {pred['path']}")
        if not np.array_equal(public_subjects, pred["public_subjects"]) or not np.array_equal(public_trial_ids, pred["public_trial_ids"]):
            raise ValueError(f"public order mismatch: {pred['path']}")

    raw_contest = np.vstack([pred["contest_oof"] for pred in preds]).T.astype(np.float32)  # type: ignore[index]
    raw_public = np.vstack([pred["public_probs"] for pred in preds]).T.astype(np.float32)  # type: ignore[index]
    rank_contest = np.vstack([rank_by_subject(raw_contest[:, idx], contest_subjects) for idx in range(raw_contest.shape[1])]).T
    rank_public = np.vstack([rank_by_subject(raw_public[:, idx], public_subjects) for idx in range(raw_public.shape[1])]).T

    dep_mask = is_dep_subject(contest_subjects)
    hc_mask = ~dep_mask
    dep_prior = float(dep_mask.mean())
    gate_alphas = [float(part.strip()) for part in args.gate_alphas.split(",") if part.strip()]
    score_spaces = [part.strip() for part in args.score_spaces.split(",") if part.strip()]

    search_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None
    best_payload: dict[str, object] | None = None

    for score_space in score_spaces:
        if score_space == "raw":
            contest_matrix = raw_contest
            public_matrix = raw_public
        elif score_space == "rank":
            contest_matrix = rank_contest
            public_matrix = rank_public
        else:
            raise ValueError(f"Unknown score space: {score_space}")

        hc_weights, hc_best, hc_rows = search_group_weights(
            contest_matrix, y, contest_subjects, hc_mask, step=args.grid_step, group_name="HC", score_space=score_space
        )
        dep_weights, dep_best, dep_rows = search_group_weights(
            contest_matrix, y, contest_subjects, dep_mask, step=args.grid_step, group_name="DEP", score_space=score_space
        )
        search_rows.extend(hc_rows)
        search_rows.extend(dep_rows)
        hc_scores = weighted_sum(contest_matrix, hc_weights)
        dep_scores = weighted_sum(contest_matrix, dep_weights)
        public_hc_scores = weighted_sum(public_matrix, hc_weights)
        public_dep_scores = weighted_sum(public_matrix, dep_weights)

        hard_oracle_scores = np.where(dep_mask, dep_scores, hc_scores).astype(np.float32)
        hard_metrics = topk_metrics(y, hard_oracle_scores, contest_subjects, 20)
        candidate_rows.append(
            {
                "score_space": score_space,
                "method": "type_oracle_not_for_public",
                "gate_alpha": None,
                "accuracy": hard_metrics["accuracy"],
                "balanced_accuracy": hard_metrics["balanced_accuracy"],
                "hc_accuracy": subset_topk_metrics(y, hard_oracle_scores, contest_subjects, hc_mask)["accuracy"],
                "dep_accuracy": subset_topk_metrics(y, hard_oracle_scores, contest_subjects, dep_mask)["accuracy"],
                "hc_weights": hc_best["weights"],
                "dep_weights": dep_best["weights"],
            }
        )

        for alpha in gate_alphas:
            effective_gate = (1.0 - alpha) * dep_prior + alpha * gate_oof
            soft_scores = ((1.0 - effective_gate) * hc_scores + effective_gate * dep_scores).astype(np.float32)
            metrics = topk_metrics(y, soft_scores, contest_subjects, 20)
            row = {
                "score_space": score_space,
                "method": "p_dep_soft",
                "gate_alpha": alpha,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "hc_accuracy": subset_topk_metrics(y, soft_scores, contest_subjects, hc_mask)["accuracy"],
                "dep_accuracy": subset_topk_metrics(y, soft_scores, contest_subjects, dep_mask)["accuracy"],
                "hc_weights": hc_best["weights"],
                "dep_weights": dep_best["weights"],
            }
            candidate_rows.append(row)
            if best_candidate is None or row["accuracy"] > float(best_candidate["accuracy"]):
                best_candidate = row
                best_payload = {
                    "score_space": score_space,
                    "hc_weights": hc_weights,
                    "dep_weights": dep_weights,
                    "contest_scores": soft_scores,
                    "public_hc_scores": public_hc_scores,
                    "public_dep_scores": public_dep_scores,
                    "alpha": alpha,
                }

    assert best_candidate is not None and best_payload is not None
    public_effective_gate = (1.0 - float(best_payload["alpha"])) * dep_prior + float(best_payload["alpha"]) * public_gate
    public_scores = (
        (1.0 - public_effective_gate) * best_payload["public_hc_scores"]
        + public_effective_gate * best_payload["public_dep_scores"]
    ).astype(np.float32)
    contest_scores = best_payload["contest_scores"].astype(np.float32)  # type: ignore[union-attr]
    contest_rank = rank_by_subject(contest_scores, contest_subjects)
    public_rank = rank_by_subject(public_scores, public_subjects)
    public_pred = topk_by_subject(public_scores, public_subjects, 4)
    public_df = pd.DataFrame(
        {
            "user_id": public_subjects,
            "trial_id": public_trial_ids,
            "score": public_scores,
            "rank_probability": public_rank,
            "p_dep_gate": public_gate,
            "p_dep_effective": public_effective_gate,
            "Emotion_label": public_pred,
        }
    )
    submission_path = output_dir / "public_test_submission_disease_aware_ensemble_top4.xlsx"
    detail_path = output_dir / "public_test_prediction_details_disease_aware_ensemble_top4.xlsx"
    public_df[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    public_df.to_excel(detail_path, index=False)
    np.savez_compressed(
        output_dir / "predictions_disease_aware_ensemble.npz",
        contest_oof=contest_rank,
        contest_y=y,
        contest_subjects=contest_subjects,
        contest_trial_ids=contest_trial_ids,
        public_probs=public_rank,
        public_subjects=public_subjects,
        public_trial_ids=public_trial_ids,
        threshold=np.array([0.5], dtype=np.float32),
        p_dep_oof=gate_oof,
        p_dep_public=public_gate,
        best_score_space=np.array([str(best_payload["score_space"])]),
        best_gate_alpha=np.array([float(best_payload["alpha"])], dtype=np.float32),
        hc_weights=np.asarray(best_payload["hc_weights"], dtype=np.float32),
        dep_weights=np.asarray(best_payload["dep_weights"], dtype=np.float32),
    )

    by_subject = by_subject_accuracy(y, contest_scores, contest_subjects, 20)
    with pd.ExcelWriter(output_dir / "disease_aware_ensemble_report.xlsx") as writer:
        pd.DataFrame(search_rows).sort_values(["group", "score_space", "accuracy"], ascending=[True, True, False]).to_excel(
            writer, sheet_name="group_weight_search", index=False
        )
        pd.DataFrame(candidate_rows).sort_values("accuracy", ascending=False).to_excel(
            writer, sheet_name="soft_candidates", index=False
        )
        by_subject.to_excel(writer, sheet_name="contest_by_subject", index=False)
        public_df.to_excel(writer, sheet_name="public_top4_details", index=False)

    report = {
        "best": best_candidate,
        "best_top20_metrics": topk_metrics(y, contest_scores, contest_subjects, 20),
        "public_positive_count": int((public_df["Emotion_label"] == 1).sum()),
        "public_neutral_count": int((public_df["Emotion_label"] == 0).sum()),
        "sources": [pred["path"] for pred in preds],
        "outputs": {
            "submission": str(submission_path),
            "details": str(detail_path),
            "predictions": str(output_dir / "predictions_disease_aware_ensemble.npz"),
            "report": str(output_dir / "disease_aware_ensemble_report.xlsx"),
        },
    }
    (output_dir / "disease_aware_ensemble_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
