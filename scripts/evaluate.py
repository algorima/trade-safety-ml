from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_safety_ml.custom_ml import CustomML
from trade_safety_ml.model import compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/custom_ml_ckpt", help="Checkpoint dir containing model.joblib")
    ap.add_argument("--csv", default="data/labeled.csv", help="Labeled CSV to evaluate on")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/eval_report.json")
    args = ap.parse_args()

    model = CustomML.load(args.ckpt)

    df = pd.read_csv(args.csv).dropna(subset=["is_scam"]).copy()
    y = df["is_scam"].astype(int).values
    X = df.drop(columns=["is_scam"])

    proba = model.predict_proba(X)
    metrics = compute_metrics(y, proba, threshold=args.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    print("✅ report saved:", str(out_path))

if __name__ == "__main__":
    main()