#!/usr/bin/env python3
"""
Resize every annotated image to a fixed square size, preserving aspect
ratio with padding (never stretching - framing is itself a target label,
so distorting geometry would corrupt it), and pack the results into a tar
archive for fast Colab session startup.

This cache is CNN input only. Classical features must not be computed from
it: downsampling to 256px suppresses the blur signal, and the black padding
skews exposure statistics and the border-edge framing feature. Classical
features read the original images instead (see docs/decisions.md).

Usage: python scripts/prepare_cache.py DATA_DIR [--size 256] [--out PATH]
"""
import argparse
import io
import sys
import tarfile
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import get_splits, image_subdir  # noqa: E402


def resize_with_padding(image, size):
    """Resize preserving aspect ratio, then letterbox-pad to a square."""
    image = ImageOps.exif_transpose(image)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    out_path = Path(args.out) if args.out else data_dir / f"image_cache_{args.size}.tar"

    splits = get_splits(data_dir)
    # val and test are a strict partition of val.json, so this has no
    # duplicate keys; every annotated image we actually use gets cached
    # exactly once (the unused official test.json images are never touched).
    all_records = {}
    for name in ("train", "val", "test"):
        for record in splits[name]:
            all_records[record["image"]] = record

    print(f"Caching {len(all_records)} images at {args.size}x{args.size} -> {out_path}")
    written = 0
    with tarfile.open(out_path, "w") as tar:
        for image_name in sorted(all_records):
            src_path = images_dir / image_subdir(image_name) / image_name
            with Image.open(src_path) as im:
                resized = resize_with_padding(im.convert("RGB"), args.size)

            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=95)
            data = buf.getvalue()

            info = tarfile.TarInfo(name=image_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            written += 1
            if written % 2000 == 0:
                print(f"  {written}/{len(all_records)}")

    print(f"Done: {written} images written to {out_path}")


if __name__ == "__main__":
    main()
