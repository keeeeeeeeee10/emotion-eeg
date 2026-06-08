from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EEGTrialDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = x.astype(np.float32)
        self.y = None if y is None else y.astype(np.int64)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).unsqueeze(0)
        if self.y is None:
            return x
        return x, torch.tensor(self.y[idx], dtype=torch.long)


class EEGNet(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int = 30,
        n_samples: int = 2500,
        n_classes: int = 2,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 125), padding=(0, 62), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 32), padding=(0, 16), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self.features(dummy)
            flat = feat.reshape(1, -1).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor, *, return_embedding: bool = False):
        emb = self.embedding(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


@dataclass
class TrainResult:
    best_state: dict[str, torch.Tensor]
    best_accuracy: float
    history: list[dict[str, float]]


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        EEGTrialDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, *, batch_size: int = 256, device: torch.device | None = None) -> np.ndarray:
    model.eval()
    device = device or get_device()
    loader = make_loader(x, None, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, *, batch_size: int = 256, device: torch.device | None = None) -> dict[str, float]:
    model.eval()
    device = device or get_device()
    loader = make_loader(x, y, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        total_loss += float(criterion(logits, yb).item())
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    return {"loss": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


def fit_supervised(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: torch.device | None = None,
) -> TrainResult:
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        accs: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            accs.append(accuracy_from_logits(logits.detach(), yb))
        train_loss = float(np.mean(losses))
        train_acc = float(np.mean(accs))
        row = {"epoch": float(epoch), "train_loss": train_loss, "train_accuracy": train_acc}
        if x_val is not None and y_val is not None:
            val = evaluate(model, x_val, y_val, batch_size=batch_size, device=device)
            row.update({"val_loss": val["loss"], "val_accuracy": val["accuracy"]})
            score = val["accuracy"]
        else:
            score = train_acc
        if score > best_acc:
            best_acc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if k != "epoch" else f"epoch={int(v):03d}" for k, v in row.items()), flush=True)
    model.load_state_dict(best_state)
    return TrainResult(best_state=best_state, best_accuracy=best_acc, history=history)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    ns = max(source.shape[0] - 1, 1)
    nt = max(target.shape[0] - 1, 1)
    cs = source.t().matmul(source) / ns
    ct = target.t().matmul(target) / nt
    return torch.mean((cs - ct) ** 2)


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((source.mean(dim=0) - target.mean(dim=0)) ** 2)

