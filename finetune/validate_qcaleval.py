#!/usr/bin/env python3
"""Validate QcalEval JSONL files before PV tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PV_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PV_DIR / "QcalEval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--files",
        nargs="+",
        default=["qcaleval_zs_sft.jsonl", "qcaleval_icl_sft.jsonl"],
    )
    parser.add_argument("--allow_missing_images", action="store_true")
    return parser.parse_args()


def image_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def main() -> None:
    args = parse_args()
    total_rows = 0
    total_images = 0
    missing_images = 0
    placeholder_mismatch = 0

    for filename in args.files:
        path = args.data_dir / filename
        rows = 0
        images = 0
        missing = 0
        mismatch = 0

        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                rows += 1
                record = json.loads(line)
                conversations = record.get("conversations", [])
                if len(conversations) != 2:
                    raise ValueError(f"{filename}:{line_number} expected 2 conversation turns")
                if conversations[0].get("role") != "user" or conversations[1].get("role") != "assistant":
                    raise ValueError(f"{filename}:{line_number} expected user/assistant roles")

                image_paths = image_list(record.get("image"))
                images += len(image_paths)
                placeholders = conversations[0].get("content", "").count("<image>")
                if placeholders and placeholders != len(image_paths):
                    mismatch += 1
                for rel_path in image_paths:
                    full_path = Path(rel_path)
                    if not full_path.is_absolute():
                        full_path = args.data_dir / full_path
                    if not full_path.exists():
                        missing += 1

        total_rows += rows
        total_images += images
        missing_images += missing
        placeholder_mismatch += mismatch
        print(
            f"{filename}: rows={rows}, image_refs={images}, "
            f"missing_images={missing}, placeholder_mismatch_rows={mismatch}"
        )

    print(
        f"TOTAL: rows={total_rows}, image_refs={total_images}, "
        f"missing_images={missing_images}, placeholder_mismatch_rows={placeholder_mismatch}"
    )
    if missing_images and not args.allow_missing_images:
        raise SystemExit(
            "Missing image files found. Copy the referenced images under QcalEval/images "
            "or rerun with --allow_missing_images if you only need a script plumbing check."
        )


if __name__ == "__main__":
    main()
