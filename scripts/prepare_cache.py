#!/usr/bin/env python3
"""
Build the CNN image cache: every image of the requested partitions
letterboxed to a fixed square (src/data/cache.py), stored as JPEG in tar
shards of SHARD_SIZE images, plus one manifest row per image.

Originals are read directly from DATA_DIR/images/{train,val}.zip (no unzip
needed), selected from the annotation records, never by listing folders.

This cache is CNN input only. Classical features read the original images
(docs/decisions.md): the padding bars would skew exposure statistics and the
border-edge framing feature, and JPEG re-encoding alters fine texture.

Resumable: each shard is built in a local work directory and copied to
OUT_DIR only when complete (written as .part, then renamed). A rerun skips
every shard already complete in OUT_DIR, so an interrupted build loses at
most the shard in progress. manifest.csv is assembled from the per-shard
CSVs at the end.

The test partition is excluded by default (it is not read during
development); the final evaluation run builds it with --partitions test.

Usage:
  python -u scripts/prepare_cache.py DATA_DIR [--size 224] [--partitions train,val]
      [--shard-size 2000] [--out-dir DIR] [--work-dir DIR]
"""
import argparse
import csv
import io
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.cache import (  # noqa: E402
    CACHE_SIZE,
    JPEG_QUALITY,
    MANIFEST_COLUMNS,
    SHARD_SIZE,
    letterbox,
    plan_shards,
    shard_name,
)
from src.data.splits import get_splits, image_subdir  # noqa: E402
from src.eda.image_props import open_archive, zip_member_index  # noqa: E402
from src.features.classical import read_image  # noqa: E402

PROGRESS_INTERVAL_SEC = 30


def shard_complete(out_dir, name, expected_names):
    """A shard counts as done only if its tar and CSV both exist and the tar
    holds exactly the expected images."""
    tar_path, csv_path = out_dir / f"{name}.tar", out_dir / f"{name}.csv"
    if not (tar_path.exists() and csv_path.exists()):
        return False
    try:
        with tarfile.open(tar_path) as tar:
            return sorted(tar.getnames()) == sorted(expected_names)
    except (tarfile.TarError, OSError):
        return False


def publish(src, dst):
    """Copy to dst via a .part file, so a partial copy is never mistaken for
    a finished shard."""
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    part.replace(dst)


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--size", type=int, default=CACHE_SIZE)
    parser.add_argument("--partitions", default="train,val")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="local scratch for building shards (default: a temp dir)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir / f"image_cache_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="cache_build_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    splits = get_splits(args.data_dir)
    unknown = [p for p in partitions if p not in splits]
    if unknown:
        parser.error(f"unknown partitions: {unknown}")
    partition_of = {r["image"]: p for p in partitions for r in splits[p]}
    shards = plan_shards(partition_of, args.shard_size)
    total = len(partition_of)

    done_shards = {i for i, names in enumerate(shards)
                   if shard_complete(out_dir, shard_name(i), names)}
    already = sum(len(shards[i]) for i in done_shards)
    print(f"[cache] {total:,} images ({', '.join(partitions)}) -> {len(shards)} shards "
          f"at {args.size}px in {out_dir}")
    print(f"[cache] {len(done_shards)} shards ({already:,} images) already complete; "
          f"{total - already:,} images to build")

    archives, indexes = {}, {}
    try:
        for subdir in sorted({image_subdir(n) for n in partition_of}):
            archives[subdir] = open_archive(args.data_dir, subdir)
            indexes[subdir] = zip_member_index(archives[subdir])
        missing = [n for n in partition_of if n not in indexes[image_subdir(n)]]
        if missing:
            sys.exit(f"[cache] {len(missing)} annotated images missing from the zips, "
                     f"e.g. {missing[:3]}")

        start = last_report = time.time()
        built = 0
        for i, names in enumerate(shards):
            name = shard_name(i)
            if i in done_shards:
                continue
            tar_tmp, csv_tmp = work_dir / f"{name}.tar", work_dir / f"{name}.csv"
            with tarfile.open(tar_tmp, "w") as tar, \
                    open(csv_tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                for image_name in names:
                    subdir = image_subdir(image_name)
                    raw = archives[subdir].read(indexes[subdir][image_name])
                    image = read_image(io.BytesIO(raw))
                    square, placement = letterbox(image, args.size)
                    buf = io.BytesIO()
                    square.save(buf, format="JPEG", quality=JPEG_QUALITY)
                    info = tarfile.TarInfo(name=image_name)
                    info.size = buf.tell()
                    buf.seek(0)
                    tar.addfile(info, buf)
                    writer.writerow({"image": image_name, "partition": partition_of[image_name],
                                     "shard": name, "native_width": image.width,
                                     "native_height": image.height, **placement})
                    built += 1

                    now = time.time()
                    if now - last_report >= PROGRESS_INTERVAL_SEC:
                        done = already + built
                        rate = built / (now - start)
                        filled = int(20 * done / total)
                        print(f"[{'#' * filled}{'-' * (20 - filled)}] {done / total:4.0%}  "
                              f"{done:,}/{total:,} | {rate:5.1f} img/s | "
                              f"elapsed {fmt_time(now - start)} | "
                              f"ETA {fmt_time((total - done) / rate)}")
                        last_report = now
            publish(tar_tmp, out_dir / f"{name}.tar")
            publish(csv_tmp, out_dir / f"{name}.csv")
            tar_tmp.unlink()
            csv_tmp.unlink()
            print(f"{name} done ({len(names):,} images) -> {out_dir}")
    finally:
        for archive in archives.values():
            archive.close()

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for i in range(len(shards)):
            with open(out_dir / f"{shard_name(i)}.csv", newline="", encoding="utf-8") as f:
                writer.writerows(csv.DictReader(f))
    print(f"[cache] DONE: {total:,} images in {len(shards)} shards, "
          f"{built:,} built this run, manifest {manifest}")


if __name__ == "__main__":
    main()
