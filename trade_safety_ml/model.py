from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, log_loss, confusion_matrix, mean_absolute_error, mean_squared_error
)

# sklearn 1.4+ provides root_mean_squared_error; squared= arg removed from mean_squared_error in 1.6+
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:  # pragma: no cover
    root_mean_squared_error = None

from xgboost import XGBClassifier

from .features import parse_trades, has_image, text_signals, build_text

DERIVED_NUMERIC_COLS = [
    "trades_count", "score", "comment_count", "has_image",
    "kw_urgent","kw_external_contact","kw_payment_first","kw_paypal_gs","kw_timestamp","kw_shipping","kw_price"
]
CATEGORICAL_COLS = ["transaction_type","country","flair"]
TEXT_COL = "text"

def flatten_text_for_tfidf(df):
    """Pickle-safe text selector for ColumnTransformer."""
    if isinstance(df, pd.DataFrame):
        # df has a single column (TEXT_COL)
        s = df.iloc[:, 0]
    else:
        s = pd.Series(df)
    return s.astype(str).fillna("")

class DerivedFeatures(BaseEstimator, TransformerMixin):
    """
    라벨링 가이드 기반으로 파생 feature를 추가합니다.
      - trades_count: author_flair에서 'Trades: N' 파싱
      - has_image: is_gallery / first_image_url
      - keyword signals: urgent, 외부연락, 선입금, PayPal G&S, timestamp/proof, shipping, price
      - text: title+selftext 결합 (TF-IDF 용)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()

        X["trades_count"] = X.get("author_flair").apply(parse_trades)
        X["has_image"] = [has_image(u, g) for u, g in zip(X.get("first_image_url"), X.get("is_gallery"))]

        sigs = [text_signals(t, s) for t, s in zip(X.get("title"), X.get("selftext"))]
        keys = list(sigs[0].keys()) if sigs else []
        for k in keys:
            X[k] = [d.get(k, 0) for d in sigs]

        X[TEXT_COL] = [build_text(t, s) for t, s in zip(X.get("title"), X.get("selftext"))]

        for col in ["score","comment_count"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        return X

def _make_onehot() -> OneHotEncoder:
    # scikit-learn >=1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _make_preprocessor(for_mlp: bool = False) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # keep sparse friendly
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot()),
    ])

    text_pipe = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        max_features=5000,
        min_df=1
    )

    text_selector = Pipeline(steps=[
        ("select", FunctionTransformer(flatten_text_for_tfidf, validate=False)),
        ("tfidf", text_pipe),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, DERIVED_NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
            ("txt", text_selector, [TEXT_COL]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return ct

def build_model(model_name: str, pos_weight: Optional[float] = None, random_state: int = 42):
    """
    Returns sklearn Pipeline = DerivedFeatures -> preprocess -> classifier
    model_name: 'xgb' | 'logreg' | 'mlp'
    """
    model_name = model_name.lower().strip()
    if model_name not in {"xgb","logreg","mlp"}:
        raise ValueError("model_name must be one of: xgb, logreg, mlp")

    if model_name == "mlp":
        pre = _make_preprocessor(for_mlp=True)
        clf = Pipeline(steps=[
            ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                random_state=random_state,
            ))
        ])
        return Pipeline(steps=[
            ("derive", DerivedFeatures()),
            ("pre", pre),
            ("clf", clf),
        ])

    pre = _make_preprocessor(for_mlp=False)

    if model_name == "logreg":
        clf = LogisticRegression(
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        clf = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=4,
            scale_pos_weight=float(pos_weight) if pos_weight is not None else 1.0,
        )

    return Pipeline(steps=[
        ("derive", DerivedFeatures()),
        ("pre", pre),
        ("clf", clf),
    ])

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    y_pred = (y_proba >= threshold).astype(int)
    out: Dict[str, Any] = {}

    out["n"] = int(len(y_true))
    out["pos_rate"] = float(np.mean(y_true))

    out["roc_auc"] = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None
    out["pr_auc"] = float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    out["logloss"] = float(log_loss(y_true, y_proba, labels=[0,1]))
    # probability regression-style metrics
    out["mae"] = float(mean_absolute_error(y_true, y_proba))
    mse = float(mean_squared_error(y_true, y_proba))
    out["rmse"] = float(root_mean_squared_error(y_true, y_proba)) if root_mean_squared_error is not None else float(np.sqrt(mse))

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    out["confusion_matrix"] = {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
    out["threshold"] = float(threshold)
    return out
