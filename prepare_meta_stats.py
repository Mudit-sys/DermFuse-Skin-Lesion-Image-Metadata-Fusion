"""Create inference metadata statistics from HAM10000 and the saved train split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing HAM10000_metadata.csv")
    parser.add_argument(
        "--split", type=Path,
        default=ROOT / "dermfuse_out" / "split.csv",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "meta_stats.json")
    args = parser.parse_args()

    metadata_path = args.data_dir / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Could not find {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    split = pd.read_csv(args.split)
    if not {"image_id", "split"}.issubset(split.columns):
        raise ValueError("split.csv must contain image_id and split columns.")
    data = metadata.merge(split[["image_id", "split"]], on="image_id", how="inner", validate="one_to_one")
    if len(data) != len(metadata):
        raise ValueError("The saved split does not cover every metadata row exactly once.")
    train = data[data["split"] == "train"].copy()
    if train.empty:
        raise ValueError("The saved split has no training rows.")

    data["sex"] = data["sex"].fillna("unknown").astype(str).str.lower()
    data["localization"] = data["localization"].fillna("unknown").astype(str).str.lower()
    train["age"] = pd.to_numeric(train["age"], errors="coerce")
    age_median = float(train["age"].median())
    age_filled = train["age"].fillna(age_median)
    age_mean = float(age_filled.mean())
    # pandas std() is sample standard deviation; this matches the notebook exactly.
    age_std = float(age_filled.std() + 1e-8)
    sites = sorted(data["localization"].unique().tolist())
    if "unknown" not in sites:
        sites = sorted(sites + ["unknown"])
    if len(sites) + 5 != 20:
        raise ValueError(f"Expected 15 body-site categories (20 metadata features); found {len(sites)}.")

    result = {
        "age_median": age_median,
        "age_mean": age_mean,
        "age_std": age_std,
        "sites": sites,
        "sex": ["female", "male", "unknown"],
        "meta_dim": len(sites) + 5,
        "split_hash_md5": hashlib.md5(args.split.read_bytes()).hexdigest()[:8],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; metadata dimension={result['meta_dim']}; site categories={len(sites)}")


if __name__ == "__main__":
    main()
