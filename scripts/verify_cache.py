#!/usr/bin/env python3
"""
Verify the CNN image cache built by scripts/prepare_cache.py and write
reports/cache_verification.md.

Checks: exact image set (from the annotation records, no test images, no
duplicates) across manifest and shards; every image decodes to size x size
RGB; content size equals the size implied by the native dimensions in
DATA_DIR/image_properties.csv (aspect preserved, long edge exactly `size`,
no EXIF rotation - a rotated image would have swapped dimensions); padding
is the expected colour; a sample of images rebuilt from the original zips
matches the cache within JPEG tolerance.

Usage: python scripts/verify_cache.py DATA_DIR [--cache-dir DIR] [--size 224]
           [--partitions train,val] [--report reports/cache_verification.md]
"""
import argparse
import io
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
from src.data.cache import CACHE_SIZE, PAD_RGB, expected_content_size, letterbox, plan_shards, shard_name  # noqa: E402
from src.data.splits import SEED, get_splits, image_subdir  # noqa: E402
from src.eda.image_props import open_archive, zip_member_index  # noqa: E402
from src.features.classical import read_image  # noqa: E402

PAD_TOLERANCE = 6       # max |pixel - PAD_RGB| in a padding region, JPEG noise
REBUILD_TOLERANCE = 3.0  # max mean |cached - rebuilt| per image, JPEG noise
N_REBUILD = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--size", type=int, default=CACHE_SIZE)
    parser.add_argument("--partitions", default="train,val")
    parser.add_argument("--report", type=Path, default=REPO_DIR / "reports" / "cache_verification.md")
    args = parser.parse_args()
    cache_dir = args.cache_dir or args.data_dir / f"image_cache_{args.size}"

    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    splits = get_splits(args.data_dir)
    expected = sorted(r["image"] for p in partitions for r in splits[p])
    other = {r["image"] for p in splits if p not in partitions for r in splits[p]}
    shards = plan_shards(expected)
    planned_shard = {n: shard_name(i) for i, names in enumerate(shards) for n in names}

    manifest = pd.read_csv(cache_dir / "manifest.csv")
    check("manifest holds exactly the expected images, no duplicates",
          sorted(manifest["image"]) == expected and manifest["image"].is_unique,
          f"{len(manifest):,} rows, {len(expected):,} expected")
    check(f"no images from partitions outside {partitions}", not set(manifest["image"]) & other)
    check("shard assignment matches the fixed plan",
          manifest.set_index("image")["shard"].reindex(expected).eq(
              pd.Series(planned_shard).reindex(expected)).all())

    props = pd.read_csv(args.data_dir / "image_properties.csv").set_index("image")
    m = manifest.set_index("image")
    native_ok = ((m["native_width"] == props.loc[m.index, "width"])
                 & (m["native_height"] == props.loc[m.index, "height"]))
    check("native size = image_properties.csv (no EXIF rotation)", native_ok.all(),
          f"{(~native_ok).sum()} mismatches")
    exp = [expected_content_size(w, h, args.size) for w, h in zip(m["native_width"], m["native_height"])]
    content_ok = (m["content_width"].to_numpy() == [e[0] for e in exp]) & \
                 (m["content_height"].to_numpy() == [e[1] for e in exp])
    check("content size = aspect-preserving resize of the native size", content_ok.all(),
          f"{(~content_ok).sum()} mismatches")
    long_ok = m[["content_width", "content_height"]].max(axis=1) == args.size
    check(f"long edge exactly {args.size}px (small images upscaled)", long_ok.all(),
          f"{(~long_ok).sum()} mismatches; {(m[['native_width', 'native_height']].max(axis=1) < args.size).sum()} upscaled")

    pad = np.array(PAD_RGB, dtype=np.int16)
    seen, bad_size, bad_pad, unlisted = [], 0, 0, 0
    for i in range(len(shards)):
        with tarfile.open(cache_dir / f"{shard_name(i)}.tar") as tar:
            for member in tar.getmembers():
                seen.append(member.name)
                with Image.open(tar.extractfile(member)) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.int16)
                    if im.size != (args.size, args.size) or im.mode != "RGB":
                        bad_size += 1
                        continue
                if member.name not in m.index:
                    unlisted += 1
                    continue
                row = m.loc[member.name]
                # sample the middle of each padding bar, away from the content edge
                x0, y0 = int(row.offset_x), int(row.offset_y)
                strips = []
                if x0 >= 4:
                    strips += [arr[:, : x0 // 2], arr[:, args.size - x0 // 2:]]
                if y0 >= 4:
                    strips += [arr[: y0 // 2], arr[args.size - y0 // 2:]]
                if any(np.abs(s - pad).max() > PAD_TOLERANCE for s in strips if s.size):
                    bad_pad += 1
    check("shards hold exactly the manifest images, each once",
          sorted(seen) == expected, f"{len(seen):,} files in {len(shards)} shards")
    check(f"every image decodes to {args.size}x{args.size} RGB", bad_size == 0, f"{bad_size} bad")
    check("padding bars are the ImageNet-mean colour", bad_pad == 0 and unlisted == 0,
          f"{bad_pad} images off, {unlisted} not in manifest")

    rng = random.Random(SEED)
    sample = rng.sample(expected, min(N_REBUILD, len(expected)))
    diffs = []
    for name in sample:
        with open_archive(args.data_dir, image_subdir(name)) as z:
            fresh, _ = letterbox(read_image(io.BytesIO(z.read(zip_member_index(z)[name]))), args.size)
        with tarfile.open(cache_dir / f"{planned_shard[name]}.tar") as tar:
            cached = Image.open(tar.extractfile(name)).convert("RGB")
        diffs.append(np.abs(np.asarray(fresh, float) - np.asarray(cached, float)).mean())
    check(f"{len(sample)} random images rebuilt from the zips match the cache",
          max(diffs) <= REBUILD_TOLERANCE, "mean abs diff " + ", ".join(f"{d:.2f}" for d in diffs))

    passed = sum(ok for _, ok, _ in results)
    print(f"\n=== {passed}/{len(results)} checks passed")

    counts = manifest["partition"].value_counts().reindex(partitions)
    upscaled = int((m[["native_width", "native_height"]].max(axis=1) < args.size).sum())
    size_mb = sum((cache_dir / f"{shard_name(i)}.tar").stat().st_size for i in range(len(shards))) / 1e6
    lines = [
        "# CNN image cache verification", "",
        f"Cache: `image_cache_{args.size}/`, built by `scripts/prepare_cache.py`, checked by "
        "`scripts/verify_cache.py`.", "",
        f"- {len(manifest):,} images ({', '.join(f'{p} {counts[p]:,}' for p in partitions)}) "
        f"in {len(shards)} shards, {size_mb:,.0f} MB",
        f"- {args.size}x{args.size} letterbox: long edge exactly {args.size}px, aspect preserved, "
        f"padding RGB {PAD_RGB} (ImageNet mean), JPEG quality 95, no EXIF rotation",
        f"- {upscaled} images natively smaller than {args.size}px were upscaled", "",
        "| Check | Result | Detail |", "|---|---|---|",
        *[f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in results], "",
        f"**{passed}/{len(results)} checks passed.**", "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("Report written to", args.report)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
