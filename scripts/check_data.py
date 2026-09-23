"""Audit local GSF-chi dataset files without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACMP = {
    "axial_650.xlsx": (35054, "8c4fc57a4272813029b9b0fd6ecfb5e032d412d606804b26c8a24ba88ec295f0"),
    "ecd_axial_index_split.pkl": (
        3387,
        "2d6ed485fe14bc238702006b41d699342618bbab0136eebcfc9858b6aa43aa14",
    ),
    "hct_ecd_axial.pkl": (
        1899212,
        "db1d9369e0a77b2a9cf257e5470d226d9c22c86d4b08f3fa7cbc14d16841933e",
    ),
    "hct_ecd_axial_res.pkl": (
        649819,
        "b6545e2c4a99b50926dfb0e7f6d0cfe8d898ca919176fe5303e30c9d3339ba24",
    ),
    "optical_rotation_589nm.csv": (
        13879,
        "6a70b6298209be02833f4a15030863124efc20f5a523582b8b5f9936d854e935",
    ),
}
ROTA = {
    "RotA.pkl": (1773015, "20a1f7ba8c3d30329c712600d564765e0a8eb3c493f2e0bf16d82aa42439d212"),
    "RotA.xlsx": (54154, "141ae5281c034f8d08b900454fc387556b5c8702b3e2347d7141ddbd69c4daff"),
}
CENTRAL_FILES = (
    "RS_train.pkl",
    "RS_validation.pkl",
    "RS_test.pkl",
    "ranking_train.pkl",
    "ranking_validation.pkl",
    "ranking_test.pkl",
)
CENTRAL_ROWS = {
    "RS_train.pkl": 326865,
    "RS_validation.pkl": 70099,
    "RS_test.pkl": 69719,
    "ranking_train.pkl": 234622,
    "ranking_validation.pkl": 49878,
    "ranking_test.pkl": 50571,
}
MOLECULENET_FILES = ("BBBP.csv", "bace.csv", "clintox.csv.gz", "sider.csv.gz", "SAMPL.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashed(root: Path, records: dict[str, tuple[int, str]]) -> list[str]:
    errors = []
    for name, (expected_size, expected_hash) in records.items():
        path = root / name
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            errors.append(f"size mismatch: {path} expected={expected_size} actual={actual_size}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"sha256 mismatch: {path} expected={expected_hash} actual={actual_hash}")
        else:
            print(f"ok: {path} ({actual_size} bytes)")
    return errors


def check_central(data_dir: Path) -> list[str]:
    errors = []
    try:
        import pandas as pd
    except ImportError:
        return ["pandas is required to audit central pickle row counts"]
    for name in CENTRAL_FILES:
        path = data_dir / name
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            frame = pd.read_pickle(path)
        except Exception as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        expected = CENTRAL_ROWS[name]
        if len(frame) != expected:
            errors.append(f"row mismatch: {path} expected={expected} actual={len(frame)}")
        else:
            print(f"ok: {path} ({len(frame)} rows)")
    return errors


def check_central_ecd(data_dir: Path) -> list[str]:
    errors = []
    graph = data_dir / "central_ecd_raw" / "ecd_column_charity_new_smiles.npy"
    expected_graph = (342684749, "6f04d36d99ef6b2a263f7767fe24777d299eb789dc65832847e74b5aeccd6dec")
    errors.extend(check_hashed(graph.parent, {graph.name: expected_graph}))
    spectra_root = data_dir / "central_ecd_raw" / "extracted" / "ECD"
    spectra = list(spectra_root.glob("*ECD/data/*.csv"))
    if len(spectra) != 10335:
        errors.append(
            f"spectrum count mismatch: {spectra_root} expected=10335 actual={len(spectra)}"
        )
    else:
        print(f"ok: {spectra_root} ({len(spectra)} spectrum CSV files)")
    return errors


def check_acmp(data_dir: Path) -> list[str]:
    errors = check_hashed(data_dir, ACMP)
    split_path = data_dir / "ecd_axial_index_split.pkl"
    if split_path.is_file():
        try:
            with split_path.open("rb") as handle:
                split = pickle.load(handle)
            sizes = {key: len(value) for key, value in split.items()}
            expected = {"train_index": 952, "val_index": 120, "test_index": 120}
            if sizes != expected:
                errors.append(f"ACMP split mismatch: expected={expected} actual={sizes}")
            else:
                print(f"ok: ACMP split sizes {sizes}")
        except Exception as exc:
            errors.append(f"cannot read {split_path}: {exc}")
    return errors


def check_present(root: Path, names: tuple[str, ...]) -> list[str]:
    errors = []
    for name in names:
        path = root / name
        if path.is_file():
            print(f"ok: {path}")
        else:
            errors.append(f"missing: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("core", "central", "central-ecd", "acmp", "rota", "moleculenet", "all"),
        default="core",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    selected = (
        {"central", "central-ecd", "acmp", "rota", "moleculenet"}
        if args.dataset == "all"
        else ({"central", "central-ecd", "acmp"} if args.dataset == "core" else {args.dataset})
    )
    errors: list[str] = []
    if "central" in selected:
        errors.extend(check_central(args.data_dir))
    if "central-ecd" in selected:
        errors.extend(check_central_ecd(args.data_dir))
    if "acmp" in selected:
        errors.extend(check_acmp(args.data_dir))
    if "rota" in selected:
        errors.extend(check_hashed(args.data_dir / "external" / "rota", ROTA))
    if "moleculenet" in selected:
        errors.extend(check_present(args.data_dir / "moleculenet", MOLECULENET_FILES))
    if errors:
        print("\nDataset audit failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nDataset audit passed.")


if __name__ == "__main__":
    main()
