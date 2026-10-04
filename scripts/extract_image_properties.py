#!/usr/bin/env python3
"""
Record native width, height, EXIF orientation and file size for every
annotated image in the requested partitions, read from the original
train/val zip archives (DATA_DIR/images/{train,val}.zip) without extracting
them. Headers only - pixels are never decoded.

Writes DATA_DIR/image_properties.csv (data, not committed). Resumable: rows
are appended in chunks, and a rerun skips images already in the CSV.

The test partition is excluded by default: it is not read during
development (see Build plan, engineering rules).

Usage: python scripts/extract_image_properties.py DATA_DIR [--partitions train,val]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import get_splits  # noqa: E402
from src.eda.image_props import (  # noqa: E402
    PROPERTY_COLUMNS,
    open_archive,
    read_header,
    records_by_archive,
    zip_member_index,
)

OUTPUT_NAME = "image_properties.csv"
CHUNK_SIZE = 1000
PROGRESS_INTERVAL_SEC = 0.5


def show_progress(done, total, current, start):
    # Single line rewritten in place (\r) on stderr, so notebook output isn't
    # flooded with one line per image.
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    sys.stderr.write(
        f"\r[{done:>6}/{total}] {done / total * 100:5.1f}% | {rate:7.1f} img/s | "
        f"elapsed {elapsed / 60:5.1f} min | ETA {eta / 60:5.1f} min | {current}  "
    )
    sys.stderr.flush()


def is_complete(row):
    if len(row) != len(PROPERTY_COLUMNS):
        return False
    try:
        [int(v) for v in row[2:]]
    except ValueError:
        return False
    return True


def recover_existing(out_path):
    """Image names already recorded by an earlier run. The file is rewritten
    keeping only complete rows, because a killed run can leave it empty,
    headerless, or ending in a half-written line."""
    if not out_path.exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row != PROPERTY_COLUMNS and is_complete(row)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PROPERTY_COLUMNS)
        writer.writerows(rows)
    return {row[0] for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--partitions", default="train,val",
                        help="comma-separated partitions from get_splits() (default: train,val)")
    args = parser.parse_args()

    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    splits = get_splits(args.data_dir)
    unknown = [p for p in partitions if p not in splits]
    if unknown:
        parser.error(f"unknown partitions: {unknown}")

    out_path = args.data_dir / OUTPUT_NAME
    done_names = recover_existing(out_path)
    todo = [
        (name, record)
        for name in partitions
        for record in splits[name]
        if record["image"] not in done_names
    ]
    total = len(todo)
    print(f"{len(done_names)} images already in {out_path}; {total} to read", file=sys.stderr)
    if total == 0:
        return

    partition_of = {record["image"]: name for name, record in todo}
    start = last_update = time.time()
    done = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(PROPERTY_COLUMNS)
        for subdir, records in records_by_archive([r for _, r in todo]).items():
            with open_archive(args.data_dir, subdir) as archive:
                index = zip_member_index(archive)
                missing = [r["image"] for r in records if r["image"] not in index]
                if missing:
                    sys.exit(f"\n{len(missing)} annotated images missing from {subdir}.zip, "
                             f"e.g. {missing[:3]}")
                for record in records:
                    name = record["image"]
                    width, height, orientation, file_bytes = read_header(archive, index[name])
                    writer.writerow([name, partition_of[name], width, height,
                                     orientation, file_bytes])
                    done += 1
                    if done % CHUNK_SIZE == 0:
                        f.flush()
                    now = time.time()
                    if now - last_update >= PROGRESS_INTERVAL_SEC or done == total:
                        show_progress(done, total, name, start)
                        last_update = now
    sys.stderr.write("\n")
    print(f"Wrote {done} rows to {out_path}")


if __name__ == "__main__":
    main()
