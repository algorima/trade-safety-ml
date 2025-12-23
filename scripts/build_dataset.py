from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path

from trade_safety_ml.data import build_whole_dataset, labeled_only, unlabeled_only

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Paths to .zip/.xlsx/or directories containing xlsx")
    ap.add_argument("--out", default="data/whole_dataset.csv", help="Output CSV path")
    ap.add_argument("--extract-dir", default="data/_extracted", help="Where to extract zip files")
    args = ap.parse_args()

    df = build_whole_dataset(args.inputs, out_csv=args.out, extract_dir=args.extract_dir)
    labeled = labeled_only(df)
    unlabeled = unlabeled_only(df)

    out_dir = Path(args.out).parent
    labeled_path = out_dir / "labeled.csv"
    unlabeled_path = out_dir / "unlabeled.csv"
    labeled.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    unlabeled.to_csv(unlabeled_path, index=False, encoding="utf-8-sig")

    print(f"✅ whole: {len(df)} rows -> {args.out}")
    print(f"✅ labeled: {len(labeled)} rows -> {labeled_path}")
    print(f"✅ unlabeled: {len(unlabeled)} rows -> {unlabeled_path}")

if __name__ == "__main__":
    main()