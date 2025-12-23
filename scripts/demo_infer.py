from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from pathlib import Path

from trade_safety_ml.custom_ml import CustomML

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/custom_ml_ckpt")
    ap.add_argument("--json", default=None, help="Path to a JSON file with a single post dict")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model = CustomML.load(args.ckpt)

    if args.json:
        post = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        # minimal example
        post = {
            "title": "[WTS][USA] Twice Photocards - PayPal G&S only",
            "selftext": "Prices: $5 each. Timestamp included. Shipping with tracking.",
            "author_flair": "Trades: 52",
            "transaction_type": "WTS",
            "country": "USA",
            "flair": "Photocard",
            "score": 5,
            "comment_count": 2,
            "first_image_url": "https://i.redd.it/xxx.jpg",
            "is_gallery": True,
        }

    out = model.analyze_post(post, threshold=args.threshold)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()