"""Build the public central-ECD cache without starting model training."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from gsf_chi.data.central_ecd import build_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "data" / "central_ecd_raw" / "extracted" / "ECD"
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=ROOT / "data" / "central_ecd_raw" / "ecd_column_charity_new_smiles.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "central_ecd_raw" / "gsf_central_ecd_cache.pkl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache after rebuilding it from the source data.",
    )
    args = parser.parse_args()
    if args.output.exists() and (not args.overwrite):
        raise SystemExit(f"refusing to overwrite existing cache: {args.output}")
    samples, split, audit = build_dataset(args.raw_root, args.graph_path)
    payload = {"samples": samples, "split": split, "audit": audit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
