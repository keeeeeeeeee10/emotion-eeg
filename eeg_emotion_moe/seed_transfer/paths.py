from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT

DEFAULT_SEED_ROOT = PROJECT_ROOT / "SEED" / "SEED_EEG"
DEFAULT_CONTEST_ROOT = PROJECT_ROOT / "赛题四数据集及说明文档"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    return Path(value).expanduser().resolve()
