from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0).astype(np.float32)
        scale = x.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


def fit_logistic_adam(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 25,
    batch_size: int = 1024,
    lr: float = 0.01,
    l2: float = 1e-4,
    seed: int = 2026,
    class_weight: bool = True,
    sample_weight: np.ndarray | None = None,
    init_model: LogisticModel | None = None,
    verbose: bool = True,
) -> tuple[LogisticModel, TrainingHistory]:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    y = y.astype(np.float32)
    n_samples, n_features = x.shape
    rng = np.random.default_rng(seed)

    if init_model is not None:
        if init_model.weights.shape[0] != n_features:
            raise ValueError(
                f"init_model has {init_model.weights.shape[0]} features, expected {n_features}"
            )
        weights = init_model.weights.astype(np.float32).copy()
        bias = float(init_model.bias)
    else:
        weights = np.zeros(n_features, dtype=np.float32)
        pos_rate = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        bias = float(np.log(pos_rate / (1.0 - pos_rate)))

    base_weight = np.ones(n_samples, dtype=np.float32)
    if sample_weight is not None:
        base_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if base_weight.shape[0] != n_samples:
            raise ValueError(f"sample_weight length {base_weight.shape[0]} != {n_samples}")
    if class_weight:
        pos_count = max(float(y.sum()), 1.0)
        neg_count = max(float(n_samples - y.sum()), 1.0)
        class_weights = np.where(y > 0.5, n_samples / (2.0 * pos_count), n_samples / (2.0 * neg_count)).astype(np.float32)
        sample_weight = base_weight * class_weights
    else:
        sample_weight = base_weight

    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = 0.0
    vb = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    losses: list[float] = []
    accuracies: list[float] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            wb = sample_weight[idx]
            pred = sigmoid(xb @ weights + bias).astype(np.float32)
            diff = (pred - yb) * wb
            norm = max(float(wb.sum()), 1.0)
            grad_w = (xb.T @ diff) / norm + l2 * weights
            grad_b = float(diff.sum() / norm)

            step += 1
            mw = beta1 * mw + (1.0 - beta1) * grad_w
            vw = beta2 * vw + (1.0 - beta2) * (grad_w * grad_w)
            mb = beta1 * mb + (1.0 - beta1) * grad_b
            vb = beta2 * vb + (1.0 - beta2) * (grad_b * grad_b)
            mw_hat = mw / (1.0 - beta1**step)
            vw_hat = vw / (1.0 - beta2**step)
            mb_hat = mb / (1.0 - beta1**step)
            vb_hat = vb / (1.0 - beta2**step)
            weights -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
            bias -= lr * mb_hat / (float(np.sqrt(vb_hat)) + eps)

        probs = sigmoid(x @ weights + bias)
        loss_terms = -(y * np.log(probs + 1e-8) + (1.0 - y) * np.log(1.0 - probs + 1e-8))
        loss = float((loss_terms * sample_weight).mean() + 0.5 * l2 * np.sum(weights * weights))
        acc = float(((probs >= 0.5).astype(np.float32) == y).mean())
        losses.append(loss)
        accuracies.append(acc)
        if verbose:
            print(f"epoch={epoch:03d} loss={loss:.6f} acc={acc:.4f}")

    return LogisticModel(weights=weights.astype(np.float32), bias=float(bias)), TrainingHistory(losses, accuracies)


@dataclass
class SavedPipeline:
    standardizer: Standardizer
    model: LogisticModel
    feature_name: str
    window_seconds: float
    channel_names: list[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.transform(x))

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)


def save_pipeline(path: Path, pipeline: SavedPipeline) -> None:
    np.savez_compressed(
        path,
        mean=pipeline.standardizer.mean_,
        scale=pipeline.standardizer.scale_,
        weights=pipeline.model.weights,
        bias=np.array([pipeline.model.bias], dtype=np.float32),
        feature_name=np.array([pipeline.feature_name]),
        window_seconds=np.array([pipeline.window_seconds], dtype=np.float32),
        channel_names=np.array(pipeline.channel_names),
    )


def load_pipeline(path: Path) -> SavedPipeline:
    data = np.load(path, allow_pickle=False)
    standardizer = Standardizer(mean_=data["mean"].astype(np.float32), scale_=data["scale"].astype(np.float32))
    model = LogisticModel(weights=data["weights"].astype(np.float32), bias=float(data["bias"][0]))
    return SavedPipeline(
        standardizer=standardizer,
        model=model,
        feature_name=str(data["feature_name"][0]),
        window_seconds=float(data["window_seconds"][0]),
        channel_names=[str(x) for x in data["channel_names"].tolist()],
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = max(len(y_true), 1)
    out = {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    if y_prob is not None:
        out["mean_prob"] = float(np.mean(y_prob))
    return out


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (y_prob >= float(threshold)).astype(np.int64)
        metrics = binary_metrics(y_true, pred, y_prob)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
    return rows


def best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
) -> dict[str, float]:
    rows = threshold_sweep(y_true, y_prob)
    return max(rows, key=lambda row: (row[metric], row["accuracy"]))
