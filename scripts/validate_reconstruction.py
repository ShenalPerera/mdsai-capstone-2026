#!/usr/bin/env python3
"""
Validate that per-flaw vote counts can be reconstructed from the raw
per-worker CSV (VizWiz_quality_issues_train_val_test.csv) by comparing
against the published train.json / val.json vote counts.

We don't know what REJECT means in the CSV, so both interpretations are
tried and reported side by side: excluding REJECT=1 rows, and including
every row. Whichever variant reproduces the published counts exactly is
the correct interpretation.

Usage: python scripts/validate_reconstruction.py DATA_DIR
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FLAW_CODES = ["BLR", "BRT", "DRK", "FRM", "NON", "OBS", "OTH", "ROT"]
CHECK_SPLITS = ["train", "val"]
IMAGE_ID_RE = re.compile(r"(\d+)(?=\.\w+$)")

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def canonical_id(image_name):
    """The raw CSV zero-pads image IDs to 12 digits (e.g. 000000020000)
    while the published JSON / on-disk filenames use 8 digits (00022585) -
    same photo, different string. Join on the numeric ID instead of the
    literal filename so these actually match."""
    if not image_name:
        return None
    match = IMAGE_ID_RE.search(image_name)
    if not match:
        return None
    return int(match.group(1))


def load_published(data_dir, split):
    path = data_dir / "annotations" / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    published = {}
    for rec in records:
        flaws = rec.get("flaws", {})
        image_id = canonical_id(rec["image"])
        published[image_id] = {
            "image": rec["image"],
            "flaws": {code: int(flaws.get(code, 0)) for code in FLAW_CODES},
        }
    return published


def find_csv(data_dir):
    ann_dir = data_dir / "annotations"
    search_root = ann_dir if ann_dir.is_dir() else data_dir
    candidates = sorted(search_root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under {search_root}")
    return candidates[0]


def load_csv_rows(path):
    # utf-8-sig: this CSV ships with a UTF-8 BOM, which would otherwise
    # attach itself to the first header ("IMG" -> "﻿IMG") and silently
    # break every row lookup.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def reconstruct(rows, split, exclude_rejected):
    """Group raw per-worker rows by image ID and sum each flaw column."""
    totals = defaultdict(lambda: {code: 0 for code in FLAW_CODES})
    split_upper = split.upper()
    for row in rows:
        row_split = (row.get("SPLIT") or "").strip().upper()
        if row_split != split_upper:
            continue
        if exclude_rejected and str(row.get("REJECT", "0")).strip() == "1":
            continue
        image_id = canonical_id(row.get("IMG"))
        if image_id is None:
            continue
        for code in FLAW_CODES:
            value = row.get(code)
            if value not in (None, ""):
                totals[image_id][code] += int(value)
    return totals


def compare(published, reconstructed):
    exact = 0
    mismatches = []
    for image_id, pub in published.items():
        pub_flaws = pub["flaws"]
        rec = reconstructed.get(image_id)
        if rec is None:
            mismatches.append((pub["image"], "ALL", pub_flaws, "no CSV rows found"))
            continue
        if pub_flaws == rec:
            exact += 1
        else:
            diff_codes = [c for c in FLAW_CODES if pub_flaws[c] != rec[c]]
            mismatches.append((pub["image"], ",".join(diff_codes), pub_flaws, rec))
    return exact, mismatches


def run_split(data_dir, split, csv_rows):
    log(f"\n## Split: {split}")
    published = load_published(data_dir, split)
    log(f"- published records: {len(published)}")

    for variant_name, exclude_rejected in [
        ("excluding REJECT=1 rows", True),
        ("including all rows", False),
    ]:
        reconstructed = reconstruct(csv_rows, split, exclude_rejected)
        exact, mismatches = compare(published, reconstructed)
        total = len(published)
        log(f"\n### Variant: {variant_name}")
        log(f"- exact matches: {exact} / {total} ({exact / total * 100:.2f}%)")
        log(f"- mismatches: {len(mismatches)}")
        if mismatches:
            log("\n| image | mismatched flaws | published | reconstructed |")
            log("|-------|-------------------|-----------|----------------|")
            for image, codes, pub, rec in mismatches[:5]:
                log(f"| {image} | {codes} | {pub} | {rec} |")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_reconstruction.py DATA_DIR", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"DATA_DIR does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    log("# Test-Label Reconstruction Validation")
    log(f"\nDATA_DIR: `{data_dir}`")

    csv_path = find_csv(data_dir)
    log(f"\nCSV source: `{csv_path}`")
    csv_rows = load_csv_rows(csv_path)
    log(f"CSV rows: {len(csv_rows)}")

    for split in CHECK_SPLITS:
        run_split(data_dir, split, csv_rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "reconstruction_validation.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
