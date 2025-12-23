from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score

from trade_safety_ml.model import build_model, compute_metrics

def _pr_auc_scorer():
    """PR-AUC scorer compatible with scikit-learn >= 1.4.

    NOTE: scikit-learn 1.4+ replaced `needs_proba` with `response_method`.
    """
    import numpy as np

    def _ap(y_true, y_score, **_kwargs):
        y_score = np.asarray(y_score)
        # Guard: if a CV fold has only one class, AP is undefined; return 0.0 to avoid scoring crash.
        if len(np.unique(y_true)) < 2:
            return 0.0
        # Some estimators return (n_samples, 2); AP expects 1D score for binary.
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        return average_precision_score(y_true, y_score)

    return make_scorer(_ap, response_method="predict_proba", greater_is_better=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/labeled.csv", help="Labeled CSV created by scripts/build_dataset.py")
    ap.add_argument("--out", default="models/custom_ml_ckpt", help="Output checkpoint directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--candidates", nargs="+", default=["xgb","logreg"], help="Model candidates: xgb logreg mlp")
    ap.add_argument("--cv", type=int, default=5, help="Stratified K-folds (on train split) for hyperparam search")
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearch iterations (per candidate)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["is_scam"]).copy()
    df["is_scam"] = df["is_scam"].astype(int)

    y = df["is_scam"].values
    X = df.drop(columns=["is_scam"])

    # split: train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - args.train_ratio), stratify=y, random_state=args.seed
    )

    # split temp into val/test equally (by ratio)
    # proportion of val among temp:
    val_frac = args.val_ratio / (args.val_ratio + args.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_frac), stratify=y_temp, random_state=args.seed
    )

    # scale_pos_weight for xgb
    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    pos_weight = neg / pos

    print(f"Train/Val/Test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)} | pos_rate train={y_train.mean():.3f} | pos_weight={pos_weight:.2f}")

    # Hyperparam search on train split with stratified CV
    best = None
    best_score = -1.0
    best_name = None
    best_search = None

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    scorer = _pr_auc_scorer()

    for name in args.candidates:
        name = name.lower().strip()
        model = build_model(name, pos_weight=pos_weight, random_state=args.seed)

        param_dist = {}
        if name == "xgb":
            param_dist = {
                "clf__n_estimators": [200, 400, 600, 900],
                "clf__max_depth": [3, 4, 5, 6],
                "clf__learning_rate": [0.02, 0.05, 0.1],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0],
                "clf__reg_lambda": [0.5, 1.0, 2.0],
                "clf__min_child_weight": [1.0, 5.0, 10.0],
            }
        elif name == "logreg":
            param_dist = {
                "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            }
        elif name == "mlp":
            param_dist = {
                "clf__mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "clf__mlp__learning_rate_init": [5e-4, 1e-3, 2e-3],
            }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=min(args.n_iter, max(1, len(list(param_dist.values())[0])) if param_dist else args.n_iter),
            scoring=scorer,
            cv=skf,
            verbose=1,
            random_state=args.seed,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        cv_best = float(search.best_score_)
        print(f"[{name}] best CV PR-AUC = {cv_best:.4f}")

        if cv_best > best_score:
            best_score = cv_best
            best = search.best_estimator_
            best_name = name
            best_search = search

    if best is None:
        raise RuntimeError(
            "No valid model was selected. This usually happens when CV scoring failed "
            "(e.g., incompatible scorer API or folds with a single class). "
            "Fix the scorer (use response_method=\"predict_proba\") and ensure "
            "your labeled data contains both classes."
        )

    # Evaluate on val
    y_val_proba = best.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_proba, threshold=args.threshold)

    # Train final on train+val (common practice)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    # rebuild best model with best params, refit on train+val
    final_model = best
    final_model.fit(X_trainval, y_trainval)

    # Evaluate on test
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_proba, threshold=args.threshold)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, out_dir / "model.joblib")

    metadata = {
        "model_name": best_name,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "pos_weight_train": float(pos_weight),
        "threshold": float(args.threshold),
        "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "best_cv_pr_auc": float(best_score),
        "best_params": best_search.best_params_ if best_search is not None else {},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics_val.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save split CSVs (handy for debugging/repro)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    pd.concat([X_train, pd.Series(y_train, name="is_scam")], axis=1).to_csv(splits_dir/"train.csv", index=False, encoding="utf-8-sig")
    pd.concat([X_val, pd.Series(y_val, name="is_scam")], axis=1).to_csv(splits_dir/"val.csv", index=False, encoding="utf-8-sig")
    pd.concat([X_test, pd.Series(y_test, name="is_scam")], axis=1).to_csv(splits_dir/"test.csv", index=False, encoding="utf-8-sig")

    print("\n✅ Saved checkpoint:", str(out_dir))
    print("VAL metrics:", json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print("TEST metrics:", json.dumps(test_metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
