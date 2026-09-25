#!/usr/bin/env python3
"""
Verify a downloaded VizWiz-QualityIssues dataset directory.

Usage: python scripts/verify_data.py DATA_DIR

Makes no assumptions about the internal layout of annotations.zip or the
schema of the eval CSV: it inspects whatever JSON/CSV files it finds
before analysing them, since that layout has not been confirmed.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

FLAW_CODES = ["BLR", "BRT", "DRK", "FRM", "NON", "OBS", "OTH", "ROT"]
SPLITS = ["train", "val", "test"]
EXPECTED_IMAGE_COUNTS = {"train": 23431, "val": 7750, "test": 8000}

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def infer_split(text):
    text_lower = text.lower()
    for split in SPLITS:
        if split in text_lower:
            return split
    return None


def discover_annotation_files(data_dir):
    ann_dir = data_dir / "annotations"
    search_root = ann_dir if ann_dir.is_dir() else data_dir
    json_files = sorted(search_root.rglob("*.json"))
    csv_files = sorted(search_root.rglob("*.csv"))
    return json_files, csv_files


def load_json_records(path):
    """Return a list of dict records from a JSON file of unknown shape."""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if isinstance(content, list):
        records = [r for r in content if isinstance(r, dict)]
        log(f"  {path.name}: JSON list, {len(records)} records")
        if records:
            log(f"    sample record keys: {sorted(records[0].keys())}")
        return records

    if isinstance(content, dict):
        # Prefer a nested list-of-dicts value (e.g. {"annotations": [...]})
        for key, value in content.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                log(f"  {path.name}: JSON dict with '{key}' -> list of {len(value)} records")
                log(f"    sample record keys: {sorted(value[0].keys())}")
                return value
        # Otherwise treat the dict as {image_name: record}
        records = []
        for image_name, rec in content.items():
            if isinstance(rec, dict):
                rec = dict(rec)
                rec.setdefault("image", image_name)
                records.append(rec)
        log(f"  {path.name}: JSON dict keyed by image, {len(records)} records")
        if records:
            log(f"    sample record keys: {sorted(records[0].keys())}")
        return records

    log(f"  {path.name}: unrecognized JSON top-level type {type(content).__name__}")
    return []


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    log(f"  {path.name}: CSV, {len(rows)} rows")
    log(f"    columns: {columns}")
    return rows


def get_image_name(rec):
    for key in ("image", "Image", "filename", "img", "image_id"):
        if key in rec and rec[key]:
            return str(rec[key])
    return None


def get_flaw_votes(rec):
    """Return {code: int_or_None} for each flaw code found in rec."""
    votes = {code: None for code in FLAW_CODES}
    nested = rec.get("flaws")
    if isinstance(nested, dict):
        for code in FLAW_CODES:
            if code in nested and nested[code] not in (None, ""):
                votes[code] = int(nested[code])
        return votes
    for code in FLAW_CODES:
        for key in (code, code.lower(), f"flaw_{code}", f"FLAW_{code}"):
            if key in rec and rec[key] not in (None, ""):
                votes[code] = int(rec[key])
                break
    return votes


def get_unrecognizable(rec):
    for key in ("unrecognizable", "Unrecognizable", "UNRECOGNIZABLE"):
        if key in rec and rec[key] not in (None, ""):
            return int(rec[key])
    return None


def has_any_flaw_label(votes):
    return any(v is not None for v in votes.values())


def collect_records_per_split(json_files, csv_files):
    """Map each discovered record to a split, using the source filename first
    and falling back to the image filename (e.g. VizWiz_train_00022585.jpg)."""
    per_split = {split: [] for split in SPLITS}
    unassigned = []

    for path in json_files:
        records = load_json_records(path)
        file_split = infer_split(path.name)
        for rec in records:
            split = file_split or infer_split(get_image_name(rec) or "")
            if split in per_split:
                per_split[split].append(rec)
            else:
                unassigned.append(rec)

    for path in csv_files:
        rows = load_csv_rows(path)
        file_split = infer_split(path.name)
        # Some datasets carry an explicit split/partition column.
        split_col = None
        if rows:
            for candidate in ("split", "Split", "partition", "dataset", "Dataset"):
                if candidate in rows[0]:
                    split_col = candidate
                    break
        for row in rows:
            if split_col and row.get(split_col):
                split = infer_split(str(row[split_col])) or str(row[split_col]).lower()
            else:
                split = file_split or infer_split(get_image_name(row) or "")
            if split in per_split:
                per_split[split].append(row)
            else:
                unassigned.append(row)

    if unassigned:
        log(f"  WARNING: {len(unassigned)} records could not be assigned to a split")

    return per_split


def list_image_filenames(data_dir, split):
    """Return the set of on-disk image filenames (basenames) for a split."""
    images_dir = data_dir / "images"
    search_root = images_dir if images_dir.is_dir() else data_dir
    names = {p.name for p in search_root.rglob(f"*{split}*.jpg")}
    names |= {p.name for p in search_root.rglob(f"*{split}*.JPG")}
    return names


def analyse_split(split, records, image_files):
    log(f"\n### Split: {split}")
    log(f"- records: {len(records)}")
    log(f"- image files on disk: {len(image_files)}")

    annotated_names = {get_image_name(r) for r in records if get_image_name(r)}
    if image_files:
        extra_files = sorted(image_files - annotated_names)
        missing_files = sorted(annotated_names - image_files)
        log(
            f"- image files with no matching annotation record: {len(extra_files)}"
            + (f" (e.g. {extra_files[:5]})" if extra_files else "")
        )
        log(
            f"- annotated records with no matching image file: {len(missing_files)}"
            + (f" (e.g. {missing_files[:5]})" if missing_files else "")
        )

    if not records:
        log("- flaw labels present: N/A (no records found)")
        return

    votes_list = [get_flaw_votes(r) for r in records]
    labels_present = any(has_any_flaw_label(v) for v in votes_list)
    log(f"- flaw labels present: {labels_present}")

    if not labels_present:
        log("- (skipping label statistics: no flaw votes found for this split)")
        return

    log(f"\n| Flaw | >=1 vote (n) | >=1 vote (%) | >=2 votes (n) | >=2 votes (%) |")
    log(f"|------|--------------|---------------|----------------|----------------|")
    n = len(records)
    for code in FLAW_CODES:
        values = [v[code] for v in votes_list if v[code] is not None]
        if not values:
            log(f"| {code} | n/a | n/a | n/a | n/a |")
            continue
        ge1 = sum(1 for x in values if x >= 1)
        ge2 = sum(1 for x in values if x >= 2)
        log(
            f"| {code} | {ge1} | {ge1 / n * 100:.1f}% | {ge2} | {ge2 / n * 100:.1f}% |"
        )

    unrec_values = [get_unrecognizable(r) for r in records]
    unrec_values = [v for v in unrec_values if v is not None]
    if unrec_values:
        ge2 = sum(1 for x in unrec_values if x >= 2)
        log(
            f"\n- unrecognizable >= 2: {ge2} / {len(unrec_values)} "
            f"({ge2 / len(unrec_values) * 100:.1f}%)"
        )
        hist = Counter(unrec_values)
        log("\n| unrecognizable votes | count |")
        log("|-----------------------|-------|")
        for grade in range(6):
            log(f"| {grade} | {hist.get(grade, 0)} |")
    else:
        log("\n- unrecognizable: not present in this split")

    flaws_per_image = [
        sum(1 for code in FLAW_CODES if v[code] is not None and v[code] >= 1)
        for v in votes_list
    ]
    hist = Counter(flaws_per_image)
    log("\n| flaws per image (>=1 vote) | count |")
    log("|------------------------------|-------|")
    for k in sorted(hist):
        log(f"| {k} | {hist[k]} |")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_data.py DATA_DIR", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"DATA_DIR does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    log("# Data Verification Report")
    log(f"\nDATA_DIR: `{data_dir}`")

    log("\n## Image file counts")
    log(
        "\nNote: files on disk can exceed the annotated count - the raw VizWiz "
        "image pool is shared across tasks and is not limited to images "
        "annotated for quality issues. See the per-split cross-check below for "
        "the number that actually matters (annotated records with no file, and "
        "vice versa)."
    )
    image_files_by_split = {split: list_image_filenames(data_dir, split) for split in SPLITS}
    log("\n| split | files on disk | expected (paper) |")
    log("|-------|----------------|-------------------|")
    for split in SPLITS:
        log(f"| {split} | {len(image_files_by_split[split])} | {EXPECTED_IMAGE_COUNTS[split]} |")

    log("\n## Discovered annotation files")
    json_files, csv_files = discover_annotation_files(data_dir)
    if not json_files and not csv_files:
        log("  No JSON or CSV files found under annotations/.")
    for path in json_files + csv_files:
        size = human_size(path.stat().st_size)
        log(f"- {path}: {size}")

    log("\n## File contents (schema inspection)")
    per_split = collect_records_per_split(json_files, csv_files)

    log("\n## Per-split analysis")
    for split in SPLITS:
        analyse_split(split, per_split[split], image_files_by_split[split])

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "data_summary.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
