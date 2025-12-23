from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import pandas as pd

from .features import normalize_label

DEFAULT_COLUMNS = [
    "post_id","title","author","author_flair","transaction_type","country","flair",
    "score","comment_count","selftext","created_timestamp","permalink",
    "first_image_url","is_gallery","is_scam","label_notes","labeled_by","labeled_at"
]

def _read_one_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Ensure expected columns exist (tolerant)
    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[DEFAULT_COLUMNS].copy()
    df["source_file"] = path.name
    df["is_scam"] = df["is_scam"].apply(normalize_label)
    return df

def collect_xlsx_paths(inputs: List[str], extract_dir: str) -> List[Path]:
    """
    inputs: list of file/dir paths.
      - if zip: extracted into extract_dir/<zip_name>/...
      - if dir: scan recursively
      - if xlsx: use directly
    """
    out: List[Path] = []
    extract_base = Path(extract_dir)
    extract_base.mkdir(parents=True, exist_ok=True)

    for p in inputs:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(str(pp))
        if pp.is_dir():
            out.extend(list(pp.rglob("*.xlsx")))
        elif pp.suffix.lower() == ".xlsx":
            out.append(pp)
        elif pp.suffix.lower() == ".zip":
            dest = extract_base / pp.stem
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(pp, "r") as zf:
                zf.extractall(dest)
            out.extend(list(dest.rglob("*.xlsx")))
        else:
            # ignore other file types
            continue

    # de-dupe by path
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq

def build_whole_dataset(
    inputs: List[str],
    out_csv: str,
    extract_dir: str = "data/_extracted",
) -> pd.DataFrame:
    paths = collect_xlsx_paths(inputs, extract_dir=extract_dir)
    if not paths:
        raise RuntimeError("No .xlsx files found from the given inputs.")

    frames = [_read_one_xlsx(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # Basic cleanups
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce")

    # Ensure unique post_id (if duplicates, keep first non-null label)
    if df["post_id"].isna().any():
        # keep rows even if missing post_id
        pass
    else:
        if df["post_id"].duplicated().any():
            # resolve conflicts deterministically
            def resolve(group: pd.DataFrame) -> pd.Series:
                # pick row with label if exists, else first
                labeled = group.dropna(subset=["is_scam"])
                row = labeled.iloc[0] if len(labeled) else group.iloc[0]
                # if conflicting labels, store note
                if group["is_scam"].dropna().nunique() > 1:
                    row = row.copy()
                    row["label_notes"] = (str(row.get("label_notes") or "") + " | WARNING: conflicting labels across sources").strip()
                return row
            df = df.groupby("post_id", as_index=False, sort=False).apply(resolve)
            # groupby/apply returns weird index in some pandas versions
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df

def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["is_scam"]).copy()

def unlabeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_scam"].isna()].copy()
