#!/usr/bin/env python3
"""Integrity checks for FACTOR-Bench.

Usage:
    python validate.py [--data data/factor_bench.json] [--image-root /path/to/coco]

With --image-root, also checks that every referenced COCO image exists on disk.
Exits non-zero if any check fails.
"""
import os
import sys
import json
import argparse

OPS = {"NOT", "AND", "OR", "BUT_NOT", "NEITHER",
       "DE_MORGAN", "DOUBLE_NEG", "COMM_AND", "COMM_OR", "COMPOUND"}
TYPES = {"operator", "equivalence", "compound"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/factor_bench.json")
    ap.add_argument("--image-root", default=None)
    args = ap.parse_args()

    S = json.load(open(args.data))["samples"]
    errs = []
    ids = set()
    for s in S:
        sid = s.get("sample_id", "<no-id>")
        if sid in ids:
            errs.append(f"{sid}: duplicate sample_id")
        ids.add(sid)
        if s.get("test_type") not in TYPES:
            errs.append(f"{sid}: bad test_type {s.get('test_type')!r}")
        if s.get("operator") not in OPS:
            errs.append(f"{sid}: bad operator {s.get('operator')!r}")
        if s.get("correct") not in ("a", "b", "both"):
            errs.append(f"{sid}: bad correct {s.get('correct')!r}")
        if s.get("test_type") == "equivalence" and s.get("correct") != "both":
            errs.append(f"{sid}: equivalence sample with correct != 'both'")
        if s.get("test_type") != "equivalence" and s.get("correct") not in ("a", "b"):
            errs.append(f"{sid}: non-equivalence sample with correct not in a/b")
        img = s.get("image", "")
        if not img.startswith("val2017/"):
            errs.append(f"{sid}: image not under val2017/ ({img})")
        else:
            stem = int(os.path.splitext(os.path.basename(img))[0])
            if s.get("image_id") != stem:
                errs.append(f"{sid}: image_id {s.get('image_id')} != filename {stem}")
        for pk in ("parse_a", "parse_b"):
            p = s.get(pk, {})
            if "concepts" not in p or "operator" not in p:
                errs.append(f"{sid}: {pk} malformed")
                continue
            for c in p["concepts"]:
                if "text" not in c or "is_negated" not in c:
                    errs.append(f"{sid}: {pk} concept malformed")
        if args.image_root and img.startswith("val2017/"):
            if not os.path.exists(os.path.join(args.image_root, img)):
                errs.append(f"{sid}: image not found on disk")

    print(f"samples: {len(S)} | unique ids: {len(ids)}")
    print(f"errors: {len(errs)}")
    for e in errs[:25]:
        print("  -", e)
    if len(errs) > 25:
        print(f"  ... and {len(errs) - 25} more")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
