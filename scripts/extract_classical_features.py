#!/usr/bin/env python3
"""
Extract classical quality features from the ORIGINAL images (never the
256px CNN cache) at one or more fixed long-edge resolutions, writing one CSV
per resolution. Each original is decoded once and resized per resolution,
so every CSV comes from the same code path and the same features.

Rows hold the image name, native dimensions, an upsampled flag, and the
features - no labels or partition column. Join on image name via
src/data/splits.py, which stays the single definition of the splits.

The test partition is excluded by default (it is never read during
development); pass --partitions train,val,test only for the final
evaluation run.

Usage:
  python scripts/extract_classical_features.py DATA_DIR \
      [--resolutions 224 512] [--partitions train,val] [--out-dir DIR]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import get_splits, image_subdir  # noqa: E402
from src.features.classical import (  # noqa: E402
    FEATURE_NAMES,
    compute_features,
    read_image,
    resize_long_edge,
)

META_COLUMNS = ["image", "native_width", "native_height", "upsampled"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[224, 512])
    parser.add_argument("--partitions", default="train,val")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    splits = get_splits(data_dir)
    records = sorted(
        (r for name in partitions for r in splits[name]), key=lambda r: r["image"]
    )

    writers, handles = {}, []
    for res in args.resolutions:
        path = out_dir / f"classical_features_{res}.csv"
        handle = open(path, "w", newline="", encoding="utf-8")
        handles.append(handle)
        writer = csv.DictWriter(handle, fieldnames=META_COLUMNS + FEATURE_NAMES)
        writer.writeheader()
        writers[res] = (writer, path)

    print(f"Extracting {len(FEATURE_NAMES)} features for {len(records)} images "
          f"({', '.join(partitions)}) at long edge {args.resolutions}")
    upsampled = {res: 0 for res in args.resolutions}
    try:
        for i, record in enumerate(records, start=1):
            name = record["image"]
            image = read_image(images_dir / image_subdir(name) / name)
            native_w, native_h = image.size
            for res in args.resolutions:
                is_upsampled = max(native_w, native_h) < res
                upsampled[res] += is_upsampled
                row = {
                    "image": name,
                    "native_width": native_w,
                    "native_height": native_h,
                    "upsampled": int(is_upsampled),
                }
                row.update(compute_features(resize_long_edge(image, res)))
                writers[res][0].writerow(row)
            if i % 2000 == 0:
                print(f"  {i}/{len(records)}")
    finally:
        for handle in handles:
            handle.close()

    for res in args.resolutions:
        print(f"{writers[res][1]}: {len(records)} rows, {upsampled[res]} upsampled")


if __name__ == "__main__":
    main()
