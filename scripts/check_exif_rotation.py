#!/usr/bin/env python3
"""
Measure whether the annotated train images carry EXIF Orientation tags, and
whether those tags relate to ROT votes. Tests the claim in docs/decisions.md
that removing EXIF auto-rotation is a no-op if no orientation tags exist.

Reads headers only (PIL opens lazily), so pixels are never decoded.

Usage: python scripts/check_exif_rotation.py DATA_DIR
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import get_splits, image_subdir  # noqa: E402

ORIENTATION_TAG = 0x0112
ORIENTATION_MEANING = {
    1: "normal",
    2: "mirrored",
    3: "rotated 180",
    4: "mirrored, rotated 180",
    5: "mirrored, rotated 90 CW",
    6: "rotated 90 CW",
    7: "mirrored, rotated 90 CCW",
    8: "rotated 90 CCW",
}
NO_EXIF = "no EXIF"
NO_TAG = "EXIF, no Orientation tag"
PROGRESS_INTERVAL_SEC = 0.5

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def orientation_category(path):
    with Image.open(path) as im:
        exif = im.getexif()
    if len(exif) == 0:
        return NO_EXIF
    value = exif.get(ORIENTATION_TAG)
    if value is None:
        return NO_TAG
    return int(value)


def show_progress(done, total, current, start):
    # Single line rewritten in place (\r) on stderr, so the report on stdout
    # stays clean and the notebook output isn't flooded with 23k lines.
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    sys.stderr.write(
        f"\r[{done:>6}/{total}] {done / total * 100:5.1f}% | {total - done:>6} left | "
        f"{rate:6.1f} img/s | elapsed {elapsed / 60:5.1f} min | ETA {eta / 60:5.1f} min | {current}  "
    )
    sys.stderr.flush()


def category_label(cat):
    if isinstance(cat, int):
        return f"{cat} ({ORIENTATION_MEANING.get(cat, 'unknown')})"
    return cat


def category_sort_key(cat):
    return (0, cat) if isinstance(cat, int) else (1, str(cat))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_exif_rotation.py DATA_DIR", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    images_dir = data_dir / "images"
    records = get_splits(data_dir)["train"]
    total = len(records)
    print(f"Reading EXIF headers for {total} annotated train images...", file=sys.stderr)

    counts = Counter()
    rot_votes = defaultdict(Counter)
    start = last_update = time.time()
    for done, record in enumerate(records, start=1):
        name = record["image"]
        cat = orientation_category(images_dir / image_subdir(name) / name)
        counts[cat] += 1
        rot_votes[cat][int(record["flaws"]["ROT"])] += 1

        now = time.time()
        if now - last_update >= PROGRESS_INTERVAL_SEC or done == total:
            show_progress(done, total, name, start)
            last_update = now
    sys.stderr.write("\n\n")
    non_identity = sum(n for cat, n in counts.items() if isinstance(cat, int) and cat != 1)

    log("# EXIF Orientation Check (annotated train images)")
    log(f"\nDATA_DIR: `{data_dir}`")
    log("\n## Summary")
    log(f"- images checked: {total}")
    log(f"- no EXIF at all: {counts[NO_EXIF]} ({counts[NO_EXIF] / total * 100:.1f}%)")
    log(f"- EXIF present, no Orientation tag: {counts[NO_TAG]} ({counts[NO_TAG] / total * 100:.1f}%)")
    log(f"- Orientation = 1 (identity): {counts[1]} ({counts[1] / total * 100:.1f}%)")
    log(f"- non-identity Orientation (2-8): {non_identity} ({non_identity / total * 100:.1f}%)")
    if non_identity == 0:
        log("\nNo image carries a non-identity Orientation tag, so EXIF auto-rotation"
            " would not change any pixel: removing it is a no-op for this dataset.")
    else:
        log("\nSome images carry a non-identity Orientation tag, so EXIF auto-rotation"
            " WOULD change their pixels. Compare ROT rates below: a much higher ROT>=2"
            " rate for tags 6/8 than for identity/untagged images indicates annotators"
            " saw raw (uncorrected) orientation.")

    log("\n## Orientation vs ROT vote count")
    log("\n| orientation | n | ROT=0 | ROT=1 | ROT=2 | ROT=3 | ROT=4 | ROT=5 | ROT>=2 rate |")
    log("|-------------|---|-------|-------|-------|-------|-------|-------|-------------|")
    for cat in sorted(counts, key=category_sort_key):
        votes = rot_votes[cat]
        n = counts[cat]
        positive = sum(votes[v] for v in range(2, 6))
        cells = " | ".join(str(votes[v]) for v in range(6))
        log(f"| {category_label(cat)} | {n} | {cells} | {positive / n * 100:.1f}% |")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "exif_rotation_check.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
