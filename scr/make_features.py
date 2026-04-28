from __future__ import annotations

from pathlib import Path

import numpy as np

from features import (
    check_feature_matrix,
    extract_de_features,
    extract_psd_features,
    extract_rich_features,
)


FEATURE_DIR = Path("output/features")


def main() -> None:
    x_path = FEATURE_DIR / "X_windows.npy"
    y_path = FEATURE_DIR / "y.npy"
    groups_path = FEATURE_DIR / "groups.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing input file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing input file: {y_path}")
    if not groups_path.exists():
        raise FileNotFoundError(f"Missing input file: {groups_path}")

    X = np.load(x_path)
    y = np.load(y_path)
    groups = np.load(groups_path)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"groups: {groups.shape}")

    print("\nRaw X checks:")
    check_feature_matrix("X_windows", X)

    print("\nExtracting PSD features...")
    X_psd = extract_psd_features(X)
    check_feature_matrix("X_psd", X_psd)

    print("\nExtracting DE features...")
    X_de = extract_de_features(X)
    check_feature_matrix("X_de", X_de)

    X_psd_de = np.concatenate([X_psd, X_de], axis=1).astype(np.float32, copy=False)
    check_feature_matrix("X_psd_de", X_psd_de)

    print("\nExtracting rich features...")
    X_rich = extract_rich_features(X)
    check_feature_matrix("X_rich", X_rich)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURE_DIR / "X_psd.npy", X_psd)
    np.save(FEATURE_DIR / "X_de.npy", X_de)
    np.save(FEATURE_DIR / "X_psd_de.npy", X_psd_de)
    np.save(FEATURE_DIR / "X_rich.npy", X_rich)

    print("\nSaved:")
    print(FEATURE_DIR / "X_psd.npy")
    print(FEATURE_DIR / "X_de.npy")
    print(FEATURE_DIR / "X_psd_de.npy")
    print(FEATURE_DIR / "X_rich.npy")


if __name__ == "__main__":
    main()
