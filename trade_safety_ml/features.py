from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

_URGENT_RE = re.compile(r"\b(urgent|asap|need gone|must go|desperate|quick sale)\b", re.I)
_EXTERNAL_CONTACT_RE = re.compile(r"\b(dm me|whatsapp|telegram|kakaotalk|kakao talk|line)\b", re.I)
_PAYMENT_FIRST_RE = re.compile(r"\b(payment first|pay before|pay first|send payment)\b", re.I)
_PAYPAL_GS_RE = re.compile(r"\b(paypal\s*(g&s|goods?\s*&\s*services)|g&s\s*only)\b", re.I)
_TIMESTAMP_RE = re.compile(r"\b(timestamp|proof|verification)\b", re.I)
_SHIPPING_RE = re.compile(r"\b(shipped|tracking|usps|stamped|tracked)\b", re.I)
_PRICE_RE = re.compile(r"(\$|usd\s*)\s*\d+(\.\d+)?|\b\d+(\.\d+)?\s*(usd|bucks)\b", re.I)

TRADES_RE = re.compile(r"trades?\s*:\s*(\d+)", re.I)

def parse_trades(author_flair: Any) -> Optional[int]:
    """
    Parse 'Trades: 52' -> 52. Returns None if missing/unparseable.
    """
    if author_flair is None:
        return None
    s = str(author_flair).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = TRADES_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # sometimes just a number
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    return None

def has_image(first_image_url: Any, is_gallery: Any) -> int:
    """
    1 if gallery or first_image_url is non-empty; else 0.
    """
    gal = str(is_gallery).strip().lower() if is_gallery is not None else ""
    if gal in {"true", "t", "1", "yes"}:
        return 1
    if first_image_url is not None:
        u = str(first_image_url).strip()
        if u and u.lower() != "nan":
            return 1
    return 0

def text_signals(title: Any, selftext: Any) -> Dict[str, int]:
    text = f"{'' if title is None else str(title)}\n{'' if selftext is None else str(selftext)}"
    return {
        "kw_urgent": int(bool(_URGENT_RE.search(text))),
        "kw_external_contact": int(bool(_EXTERNAL_CONTACT_RE.search(text))),
        "kw_payment_first": int(bool(_PAYMENT_FIRST_RE.search(text))),
        "kw_paypal_gs": int(bool(_PAYPAL_GS_RE.search(text))),
        "kw_timestamp": int(bool(_TIMESTAMP_RE.search(text))),
        "kw_shipping": int(bool(_SHIPPING_RE.search(text))),
        "kw_price": int(bool(_PRICE_RE.search(text))),
    }

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text(title: Any, selftext: Any) -> str:
    # For TF-IDF
    t = clean_text(title)
    b = clean_text(selftext)
    return (t + " " + b).strip()

def normalize_label(x: Any):
    """
    Normalize is_scam values to {0,1,None}
    Accepts bools, 0/1, 'TRUE'/'FALSE', etc.
    """
    if x is None:
        return None
    try:
        import pandas as pd  # optional
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, bool):
        return 1 if x else 0
    # numeric 0/1
    try:
        if isinstance(x, (int, float)) and not (isinstance(x, float) and (x != x)):
            if abs(float(x) - 1.0) < 1e-9:
                return 1
            if abs(float(x) - 0.0) < 1e-9:
                return 0
    except Exception:
        pass

    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y", "scam"}:
        return 1
    if s in {"false", "f", "0", "no", "n", "legit", "normal"}:
        return 0
    return None
