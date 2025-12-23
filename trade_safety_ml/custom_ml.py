from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

@dataclass
class CustomML:
    """
    Wrapper for the trained sklearn pipeline.
    - Save/load via joblib
    - Predict scam probability for a single post (dict) or DataFrame
    """
    pipeline: Any
    metadata: Dict[str, Any]

    @classmethod
    def load(cls, ckpt_dir: Union[str, Path]) -> "CustomML":
        ckpt_dir = Path(ckpt_dir)
        pipeline = joblib.load(ckpt_dir / "model.joblib")
        meta_path = ckpt_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(pipeline=pipeline, metadata=metadata)

    def predict_proba(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        # sklearn returns [n,2]
        proba = self.pipeline.predict_proba(X)[:, 1]
        return proba

    def predict(self, X: Union[pd.DataFrame, Dict[str, Any]], threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def analyze_post(self, post: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        p = float(self.predict_proba(post)[0])
        pred = int(p >= threshold)
        return {
            "label": "fraud" if pred == 1 else "legit",
            "risk_score": p,
            "threshold": threshold,
            "model": self.metadata.get("model_name"),
            "trained_at": self.metadata.get("trained_at"),
        }
