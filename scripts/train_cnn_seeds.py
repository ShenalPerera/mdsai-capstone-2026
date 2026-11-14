#!/usr/bin/env python3
"""
Run-to-run spread for Stage E: fine-tune the CNN and the hybrid again with
extra seeds, one run after another, by calling scripts/train_cnn.py.

The seed-42 runs are the arm entries and are not repeated here. These runs
write to cnn_runs/{finetune,hybrid_finetune}_seed<N>/ and
predictions/{cnn,hybrid}_finetune_seed<N>_val.csv; they are only used to
measure how much results vary between seeds, never to choose a run.

Each run is resumable, so rerunning this script after a disconnect skips
finished runs (they only rewrite their outputs) and continues the
interrupted one. Stops at the first failing run.

Usage: python -u scripts/train_cnn_seeds.py DATA_DIR --cache-dir DIR [--seeds 43,44]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train_cnn.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="43,44")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [(seed, hybrid) for seed in seeds for hybrid in (False, True)]
    start = time.time()
    for i, (seed, hybrid) in enumerate(runs, 1):
        name = f"{'hybrid' if hybrid else 'CNN'} seed {seed}"
        print(f"[seeds] run {i}/{len(runs)}: {name}", flush=True)
        cmd = [sys.executable, "-u", str(TRAIN), str(args.data_dir), "--cache-dir", str(args.cache_dir),
               "--mode", "finetune", "--seed", str(seed)] + (["--hybrid"] if hybrid else [])
        if subprocess.call(cmd) != 0:
            sys.exit(f"[seeds] FAILED at run {i}/{len(runs)} ({name}); rerun to resume")
    print(f"[seeds] DONE: {len(runs)} runs in {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
