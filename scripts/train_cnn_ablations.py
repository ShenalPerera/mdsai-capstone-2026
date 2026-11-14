#!/usr/bin/env python3
"""
Stage F ablation runs for the CNN, one after another, by calling
scripts/train_cnn.py. Each changes exactly one thing from the Stage D
fine-tune (seed 42, recipe unchanged):

  CNN, flaws >= 1 vote   --flaw-threshold 1: training flaw labels at >=1 vote;
                         val scoring and early stopping stay at >=2.
                         -> cnn_runs/finetune_votes1, predictions/cnn_finetune_votes1_val.csv
  CNN, random init       --random-init: no ImageNet weights.
                         -> cnn_runs/finetune_scratch, predictions/cnn_finetune_scratch_val.csv

The frozen-vs-fine-tuned ablation reuses the Stage D runs and needs no
training. Resumable: a rerun skips finished runs and continues the
interrupted one. Stops at the first failing run.

Usage: python -u scripts/train_cnn_ablations.py DATA_DIR --cache-dir DIR
"""
import argparse
from pathlib import Path

from train_cnn_seeds import run_queue

RUNS = [
    ("CNN, flaws >= 1 vote", ["--flaw-threshold", "1"]),
    ("CNN, random init", ["--random-init"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    run_queue(args.data_dir, args.cache_dir, RUNS, "ablations")


if __name__ == "__main__":
    main()
