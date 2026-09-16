#!/usr/bin/env python3
"""Run the packaged OncoMorph 4D multiclass model on four NIfTI volumes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from segmentation_service import predict_tumor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="OncoMorph multiclass glioma inference")
    parser.add_argument("--t1c", required=True, type=Path)
    parser.add_argument("--t1n", required=True, type=Path)
    parser.add_argument("--flair", required=True, type=Path)
    parser.add_argument("--t2w", required=True, type=Path)
    parser.add_argument("--output-mask", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = predict_tumor([args.t1c, args.t1n, args.flair, args.t2w], args.output_mask)
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.output_json:
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
