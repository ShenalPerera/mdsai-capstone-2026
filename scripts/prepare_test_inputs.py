#!/usr/bin/env python3
"""
Build the inputs of the final test evaluation, once, from the original
images of the test partition (the held-out half of val.json, which lives in
images/val.zip). Runs the existing scripts in order and stops at the first
failure:

  1. scripts/extract_image_properties.py --partitions test
       appends the test rows to DATA_DIR/image_properties.csv (use a local
       copy of DATA_DIR), then copies that file to OUT_DIR
  2. scripts/prepare_cache.py --partitions test --out-dir CACHE_DIR
       224px letterbox cache of the test images, a separate folder from the
       train+val cache
  3. scripts/extract_classical_features.py --partitions test --out-dir OUT_DIR
       classical features at 224 and 512 px, separate from the train+val files

DATA_DIR needs annotations/, image_properties.csv, images/val.zip and the
unzipped images/val/ folder (the feature script reads unzipped files).

Usage: python -u scripts/prepare_test_inputs.py DATA_DIR --out-dir DIR --cache-dir DIR
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True, help="test features + properties")
    parser.add_argument("--cache-dir", type=Path, required=True, help="test image cache")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        ("image properties", ["extract_image_properties.py", str(args.data_dir), "--partitions", "test"]),
        ("image cache", ["prepare_cache.py", str(args.data_dir), "--partitions", "test",
                         "--out-dir", str(args.cache_dir)]),
        ("classical features", ["extract_classical_features.py", str(args.data_dir), "--partitions", "test",
                                "--out-dir", str(args.out_dir)]),
    ]
    start = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"[test-inputs] step {i}/{len(steps)}: {name}", flush=True)
        if subprocess.call([sys.executable, "-u", str(SCRIPTS / cmd[0])] + cmd[1:]) != 0:
            sys.exit(f"[test-inputs] FAILED at step {i} ({name})")
        if name == "image properties":
            shutil.copyfile(args.data_dir / "image_properties.csv", args.out_dir / "image_properties.csv")
            print(f"[test-inputs] image_properties.csv -> {args.out_dir}", flush=True)
    print(f"[test-inputs] DONE in {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
