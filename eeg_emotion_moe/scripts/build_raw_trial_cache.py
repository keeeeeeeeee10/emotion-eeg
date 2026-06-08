from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed_transfer.paths import DEFAULT_CONTEST_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_SEED_ROOT, resolve_path
from seed_transfer.raw_trials import (
    build_contest_raw_trial_cache,
    build_public_raw_trial_cache,
    build_seed_raw_trial_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared raw 30x2500 trial caches.")
    parser.add_argument("--seed-root", type=str, default=None)
    parser.add_argument("--contest-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def show(path: Path) -> None:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"].item()))
    print(f"{path}:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    seed_root = resolve_path(args.seed_root, DEFAULT_SEED_ROOT)
    contest_root = resolve_path(args.contest_root, DEFAULT_CONTEST_ROOT)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)

    for path in [
        build_seed_raw_trial_cache(seed_root=seed_root, output_dir=output_dir, refresh=args.refresh),
        build_contest_raw_trial_cache(contest_root=contest_root, output_dir=output_dir, refresh=args.refresh),
        build_public_raw_trial_cache(contest_root=contest_root, output_dir=output_dir, refresh=args.refresh),
    ]:
        show(path)


if __name__ == "__main__":
    main()
