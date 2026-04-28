from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


EXPECTED_SHAPE = (30, 50000)
TARGET_KEYS = ("EEG_data_neu", "EEG_data_pos")


@dataclass
class FileCheckResult:
    path: Path
    loaded_with: str = ""
    variables: dict[str, tuple[tuple[int, ...], str]] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    transpose_needed: list[str] = field(default_factory=list)
    bad_shape: dict[str, tuple[int, ...]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and not self.missing_keys and not self.bad_shape


def _to_numpy_if_possible(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return np.asarray(value)
        except Exception:
            return None
    return None


def _load_mat_auto(mat_path: Path) -> tuple[dict[str, Any], str]:
    try:
        return loadmat(mat_path, squeeze_me=False, struct_as_record=False), "scipy.io.loadmat"
    except NotImplementedError:
        pass
    except ValueError:
        pass
    except Exception:
        # 其他异常继续尝试 h5py，确保尽量兼容
        pass

    data: dict[str, Any] = {}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            data[k] = f[k][()]
    return data, "h5py"


def _check_target_shape(arr: np.ndarray) -> tuple[str, tuple[int, ...]]:
    shape = tuple(arr.shape)
    if shape == EXPECTED_SHAPE:
        return "ok", shape
    if shape == (EXPECTED_SHAPE[1], EXPECTED_SHAPE[0]):
        return "transpose", shape
    return "bad", shape


def check_one_file(mat_path: Path) -> FileCheckResult:
    result = FileCheckResult(path=mat_path)

    try:
        data, loader_name = _load_mat_auto(mat_path)
        result.loaded_with = loader_name
    except Exception as e:
        result.errors.append(f"load failed: {type(e).__name__}: {e}")
        return result

    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = _to_numpy_if_possible(value)
        if arr is None:
            continue
        result.variables[key] = (tuple(arr.shape), str(arr.dtype))

    for key in TARGET_KEYS:
        if key not in result.variables:
            result.missing_keys.append(key)
            continue
        arr_shape = result.variables[key][0]
        dummy = np.empty(arr_shape, dtype=np.float32)
        status, shape = _check_target_shape(dummy)
        if status == "transpose":
            result.transpose_needed.append(key)
        elif status == "bad":
            result.bad_shape[key] = shape

    return result


def print_file_report(r: FileCheckResult) -> None:
    print(f"\nFile: {r.path}")
    if r.errors:
        print(f"  [ERROR] {'; '.join(r.errors)}")
        return

    print(f"  Loaded by: {r.loaded_with}")
    if not r.variables:
        print("  [WARN] no array-like variables found")
    else:
        for name in sorted(r.variables):
            shape, dtype = r.variables[name]
            print(f"  {name}: shape={shape}, dtype={dtype}")

    if r.missing_keys:
        print(f"  [MISSING] required vars: {r.missing_keys}")
    if r.transpose_needed:
        print(f"  [TRANSPOSE NEEDED] {r.transpose_needed}")
    if r.bad_shape:
        for k, shp in r.bad_shape.items():
            print(f"  [BAD SHAPE] {k}: {shp}, expected {EXPECTED_SHAPE} or {(50000, 30)}")

    if r.is_valid:
        print("  [OK] file valid for training pipeline")


def main() -> None:
    root = Path("data/train")
    mat_files = sorted(root.rglob("*.mat"))

    print(f"Scanning train mats under: {root.resolve()}")
    print(f"Found mat files: {len(mat_files)}")
    if not mat_files:
        print("No .mat files found. Please check data path.")
        return

    results: list[FileCheckResult] = []
    for p in mat_files:
        r = check_one_file(p)
        results.append(r)
        print_file_report(r)

    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]
    transpose_files = [r for r in results if r.transpose_needed]
    missing_files = [r for r in results if r.missing_keys]
    bad_shape_files = [r for r in results if r.bad_shape]
    load_failed = [r for r in results if r.errors]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Found mat files: {len(results)}")
    print(f"Valid files: {len(valid)}")
    print(f"Invalid files: {len(invalid)}")
    print(f"Need transpose: {len(transpose_files)}")
    print(f"Missing required vars: {len(missing_files)}")
    print(f"Bad shape files: {len(bad_shape_files)}")
    print(f"Load failed files: {len(load_failed)}")

    if invalid:
        print("\nInvalid file list:")
        for r in invalid:
            print(f"- {r.path}")
            if r.errors:
                print(f"    errors: {r.errors}")
            if r.missing_keys:
                print(f"    missing: {r.missing_keys}")
            if r.bad_shape:
                print(f"    bad_shape: {r.bad_shape}")


if __name__ == "__main__":
    main()
