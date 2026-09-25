#!/usr/bin/env python3
"""
Verify structural integrity of train.json / val.json / test.json: record
counts, image files exist on disk, flaw dict shape and value ranges, no
filename leaks across splits, and per-split filename ID ranges.

Usage: python scripts/verify_json_integrity.py DATA_DIR

Appends to reports/data_summary.md - run AFTER scripts/verify_data.py
(which creates that file) so this doesn't overwrite it.
"""
import json
import re
import sys
from pathlib import Path

FLAW_CODES = ["BLR", "BRT", "DRK", "FRM", "NON", "OBS", "OTH", "ROT"]
EXPECTED_COUNTS = {"train": 23431, "val": 7750, "test": 8000}
IMAGE_ID_RE = re.compile(r"(\d+)(?=\.\w+$)")

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def canonical_id(image_name):
    if not image_name:
        return None
    match = IMAGE_ID_RE.search(image_name)
    return int(match.group(1)) if match else None


def load_records(data_dir, split):
    path = data_dir / "annotations" / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_image_files(data_dir, split):
    images_dir = data_dir / "images"
    search_root = images_dir if images_dir.is_dir() else data_dir
    names = {p.name for p in search_root.rglob(f"*{split}*.jpg")}
    names |= {p.name for p in search_root.rglob(f"*{split}*.JPG")}
    return names


def is_valid_vote(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 5


def check_split(data_dir, split, expect_labels):
    log(f"\n### {split}.json")
    records = load_records(data_dir, split)
    expected = EXPECTED_COUNTS[split]
    status = "OK" if len(records) == expected else "MISMATCH"
    log(f"- record count: {len(records)} (expected {expected}) - {status}")

    image_files = list_image_files(data_dir, split)
    image_names = [rec.get("image") for rec in records]
    missing_files = [name for name in image_names if name not in image_files]
    log(
        f"- annotated images missing from disk: {len(missing_files)}"
        + (f" (e.g. {missing_files[:5]})" if missing_files else "")
    )

    if expect_labels:
        shape_errors = []
        range_errors = []
        for rec in records:
            image = rec.get("image", "<unknown>")
            flaws = rec.get("flaws")
            if not isinstance(flaws, dict) or sorted(flaws.keys()) != sorted(FLAW_CODES):
                shape_errors.append(image)
                continue
            unrec = rec.get("unrecognizable")
            if unrec is None:
                shape_errors.append(image)
                continue
            bad_values = [
                (code, flaws[code]) for code in FLAW_CODES if not is_valid_vote(flaws[code])
            ]
            if not is_valid_vote(unrec):
                bad_values.append(("unrecognizable", unrec))
            if bad_values:
                range_errors.append((image, bad_values))
        log(
            f"- records with malformed flaws/unrecognizable shape: {len(shape_errors)}"
            + (f" (e.g. {shape_errors[:5]})" if shape_errors else "")
        )
        log(
            f"- records with an out-of-range (not int 0-5) vote: {len(range_errors)}"
            + (f" (e.g. {range_errors[:5]})" if range_errors else "")
        )
    else:
        any_labeled = any(
            isinstance(rec.get("flaws"), dict) and rec.get("flaws") for rec in records
        )
        log(f"- flaw labels present: {any_labeled} (expected: False - withheld for the challenge)")

    ids = [canonical_id(name) for name in image_names]
    ids = [i for i in ids if i is not None]
    if ids:
        log(f"- filename ID range: {min(ids)}-{max(ids)} ({len(set(ids))} unique of {len(ids)} records)")

    return set(name for name in image_names if name)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_json_integrity.py DATA_DIR", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"DATA_DIR does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    log("\n---\n")
    log("# JSON Integrity Verification")
    log(f"\nDATA_DIR: `{data_dir}`")

    names_by_split = {
        "train": check_split(data_dir, "train", expect_labels=True),
        "val": check_split(data_dir, "val", expect_labels=True),
        "test": check_split(data_dir, "test", expect_labels=False),
    }

    log("\n### Cross-split leakage check")
    splits = list(names_by_split.keys())
    leaked = False
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1:]:
            overlap = names_by_split[split_a] & names_by_split[split_b]
            if overlap:
                leaked = True
                log(f"- {split_a} and {split_b} share {len(overlap)} filenames (e.g. {sorted(overlap)[:5]})")
    if not leaked:
        log("- no filename appears in more than one split")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "data_summary.md"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nAppended to {out_path}")


if __name__ == "__main__":
    main()
